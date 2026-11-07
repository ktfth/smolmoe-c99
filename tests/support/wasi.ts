/* Minimal WASI (preview1) host for running the WebAssembly build under bun.
 * Bun's own node:wasi fails on Windows, and the tests need only a small part
 * of WASI: arguments, stdio, files under preopened directories, clocks and
 * random bytes. Anything else answers ENOSYS. */
import { closeSync, constants, fstatSync, openSync, readSync, statSync, writeSync } from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";

const OK = 0;
const EACCES = 2;
const EEXIST = 20;
const EISDIR = 31;
const ENOTDIR = 54;
const EBADF = 8;
const EINVAL = 28;
const EIO = 29;
const ENOENT = 44;
const ENOSYS = 52;

const FILETYPE_CHARACTER_DEVICE = 2;
const FILETYPE_DIRECTORY = 3;
const FILETYPE_REGULAR_FILE = 4;
const OFLAGS_CREAT = 1;
const OFLAGS_DIRECTORY = 2;
const OFLAGS_EXCL = 4;
const OFLAGS_TRUNC = 8;
const FDFLAGS_APPEND = 1;
const RIGHT_FD_WRITE = 1n << 6n;
const WHENCE_SET = 0;
const WHENCE_CUR = 1;
const WHENCE_END = 2;

type Entry =
  | { kind: "stdin" }
  | { kind: "out"; fd: number }
  | { kind: "dir"; host: string; guest?: string }
  | { kind: "file"; host: string; fd: number; pos: number; append: boolean };

export type WasiOptions = {
  args: readonly string[];
  /* Guest directory name -> host directory. */
  preopens: Record<string, string>;
  env?: Record<string, string>;
  stdin?: Uint8Array;
};

/* A WASI function: numbers in, an errno out (never[] accepts any parameter list). */
type WasiCall = (...args: never[]) => number;

/* The WASI errno for a failed host file operation. */
function errnoOf(error: unknown): number {
  const code = (error as { code?: string }).code;
  return code === "ENOENT" ? ENOENT : code === "EEXIST" ? EEXIST : code === "EISDIR" ? EISDIR
    : code === "ENOTDIR" ? ENOTDIR : code === "EACCES" || code === "EPERM" ? EACCES : EIO;
}

/* writeSync may write less than asked (a pipe); retries until done. */
function writeAll(fd: number, data: Uint8Array, position: number | null): void {
  for (let done = 0; done < data.length;) {
    try {
      done += writeSync(fd, data, done, data.length - done, position === null ? null : position + done);
    } catch (error) {
      if ((error as { code?: string }).code !== "EAGAIN") throw error;
    }
  }
}

/* Thrown by proc_exit to unwind out of the module. */
class Exit {
  constructor(readonly code: number) {}
}

export class Wasi {
  private memory: WebAssembly.Memory | undefined;
  private readonly fds = new Map<number, Entry>();
  private readonly args: Uint8Array[];
  private readonly env: Uint8Array[];
  private readonly stdin: Uint8Array;
  private stdinPos = 0;
  private nextFd: number;
  private readonly encoder = new TextEncoder();
  private readonly decoder = new TextDecoder();

  constructor(options: WasiOptions) {
    this.args = options.args.map((a) => this.encoder.encode(`${a}\0`));
    this.env = Object.entries(options.env ?? {}).map(([k, v]) => this.encoder.encode(`${k}=${v}\0`));
    this.stdin = options.stdin ?? new Uint8Array(0);
    this.fds.set(0, { kind: "stdin" });
    this.fds.set(1, { kind: "out", fd: 1 });
    this.fds.set(2, { kind: "out", fd: 2 });
    let fd = 3;
    for (const [guest, host] of Object.entries(options.preopens)) this.fds.set(fd++, { kind: "dir", host: resolve(host), guest });
    this.nextFd = fd;
  }

  /* Runs the module's _start; returns its exit code. */
  run(module: WebAssembly.Module): number {
    const instance = new WebAssembly.Instance(module, { wasi_snapshot_preview1: this.imports() });
    this.memory = instance.exports.memory as WebAssembly.Memory;
    try {
      (instance.exports._start as () => void)();
      return 0;
    } catch (e) {
      if (e instanceof Exit) return e.code;
      throw e;
    } finally {
      for (const entry of this.fds.values()) if (entry.kind === "file") closeSync(entry.fd);
    }
  }

  private view(): DataView {
    return new DataView(this.memory!.buffer);
  }

  private bytes(ptr: number, len: number): Uint8Array {
    return new Uint8Array(this.memory!.buffer, ptr, len);
  }

  private string(ptr: number, len: number): string {
    return this.decoder.decode(this.bytes(ptr, len).slice());
  }

  /* (ptr, len) pairs of an iovec array. */
  private iovecs(iovs: number, count: number): Array<[number, number]> {
    const v = this.view();
    const out: Array<[number, number]> = [];
    for (let i = 0; i < count; ++i) out.push([v.getUint32(iovs + 8 * i, true), v.getUint32(iovs + 8 * i + 4, true)]);
    return out;
  }

  private writeStrings(list: Uint8Array[], ptrs: number, buf: number): number {
    const v = this.view();
    let at = buf;
    list.forEach((s, i) => {
      v.setUint32(ptrs + 4 * i, at, true);
      this.bytes(at, s.length).set(s);
      at += s.length;
    });
    return OK;
  }

  private sizes(list: Uint8Array[], count: number, size: number): number {
    const v = this.view();
    v.setUint32(count, list.length, true);
    v.setUint32(size, list.reduce((n, s) => n + s.length, 0), true);
    return OK;
  }

  private resolvePath(dirfd: number, ptr: number, len: number): string | undefined {
    const dir = this.fds.get(dirfd);
    if (!dir || dir.kind !== "dir") return undefined;
    const host = resolve(join(dir.host, this.string(ptr, len)));
    /* Nothing outside the preopened directory. */
    const inside = relative(dir.host, host);
    return inside === ".." || inside.startsWith(`..${sep}`) || isAbsolute(inside) ? undefined : host;
  }

  private filestat(buf: number, type: number, size: number): number {
    const v = this.view();
    for (let i = 0; i < 64; i += 8) v.setBigUint64(buf + i, 0n, true);
    v.setUint8(buf + 16, type);
    v.setBigUint64(buf + 24, 1n, true);
    v.setBigUint64(buf + 32, BigInt(size), true);
    return OK;
  }

  private read(entry: Entry, target: Uint8Array, position: number | null): number {
    if (entry.kind === "stdin") {
      const n = Math.min(target.length, this.stdin.length - this.stdinPos);
      target.set(this.stdin.subarray(this.stdinPos, this.stdinPos + n));
      this.stdinPos += n;
      return n;
    }
    if (entry.kind !== "file") return -1;
    return readSync(entry.fd, target, 0, target.length, position ?? entry.pos);
  }

  private imports(): Record<string, WasiCall> {
    const calls: Record<string, WasiCall> = {
      args_sizes_get: (count: number, size: number) => this.sizes(this.args, count, size),
      args_get: (ptrs: number, buf: number) => this.writeStrings(this.args, ptrs, buf),
      environ_sizes_get: (count: number, size: number) => this.sizes(this.env, count, size),
      environ_get: (ptrs: number, buf: number) => this.writeStrings(this.env, ptrs, buf),
      clock_time_get: (id: number, _precision: bigint, out: number) => {
        const ns = id === 0 ? BigInt(Date.now()) * 1_000_000n : process.hrtime.bigint();
        this.view().setBigUint64(out, ns, true);
        return OK;
      },
      clock_res_get: (_id: number, out: number) => {
        this.view().setBigUint64(out, 1000n, true);
        return OK;
      },
      random_get: (buf: number, len: number) => {
        /* getRandomValues fills at most 65536 bytes per call. */
        for (let at = 0; at < len; at += 65536) crypto.getRandomValues(this.bytes(buf + at, Math.min(65536, len - at)));
        return OK;
      },
      proc_exit: (code: number) => {
        throw new Exit(code);
      },
      sched_yield: () => OK,
      fd_prestat_get: (fd: number, buf: number) => {
        const e = this.fds.get(fd);
        if (!e || e.kind !== "dir" || e.guest === undefined) return EBADF;
        const v = this.view();
        v.setUint8(buf, 0);
        v.setUint32(buf + 4, this.encoder.encode(e.guest).length, true);
        return OK;
      },
      fd_prestat_dir_name: (fd: number, ptr: number, len: number) => {
        const e = this.fds.get(fd);
        if (!e || e.kind !== "dir" || e.guest === undefined) return EBADF;
        this.bytes(ptr, len).set(this.encoder.encode(e.guest).subarray(0, len));
        return OK;
      },
      fd_fdstat_get: (fd: number, buf: number) => {
        const e = this.fds.get(fd);
        if (!e) return EBADF;
        const v = this.view();
        const type = e.kind === "dir" ? FILETYPE_DIRECTORY : e.kind === "file" ? FILETYPE_REGULAR_FILE : FILETYPE_CHARACTER_DEVICE;
        v.setUint8(buf, type);
        v.setUint16(buf + 2, 0, true);
        v.setBigUint64(buf + 8, 0xffff_ffff_ffff_ffffn, true);
        v.setBigUint64(buf + 16, 0xffff_ffff_ffff_ffffn, true);
        return OK;
      },
      fd_fdstat_set_flags: () => OK,
      fd_filestat_get: (fd: number, buf: number) => {
        const e = this.fds.get(fd);
        if (!e) return EBADF;
        if (e.kind === "file") return this.filestat(buf, FILETYPE_REGULAR_FILE, fstatSync(e.fd).size);
        return this.filestat(buf, e.kind === "dir" ? FILETYPE_DIRECTORY : FILETYPE_CHARACTER_DEVICE, 0);
      },
      path_filestat_get: (dirfd: number, _flags: number, ptr: number, len: number, buf: number) => {
        const host = this.resolvePath(dirfd, ptr, len);
        if (!host) return ENOENT;
        try {
          const st = statSync(host);
          return this.filestat(buf, st.isDirectory() ? FILETYPE_DIRECTORY : FILETYPE_REGULAR_FILE, st.size);
        } catch {
          return ENOENT;
        }
      },
      path_open: (dirfd: number, _dirflags: number, ptr: number, len: number, oflags: number, rights: bigint,
        _inheriting: bigint, fdflags: number, out: number) => {
        const host = this.resolvePath(dirfd, ptr, len);
        if (!host) return ENOENT;
        let isDir = false;
        try {
          isDir = statSync(host).isDirectory();
        } catch {
          if (!(oflags & OFLAGS_CREAT)) return ENOENT;
        }
        if (oflags & OFLAGS_DIRECTORY && !isDir) return ENOENT;
        const fd = this.nextFd++;
        if (isDir) {
          this.fds.set(fd, { kind: "dir", host });
        } else {
          const write = (BigInt(rights) & RIGHT_FD_WRITE) !== 0n || (oflags & (OFLAGS_CREAT | OFLAGS_TRUNC)) !== 0;
          const append = (fdflags & FDFLAGS_APPEND) !== 0;
          let flags = write ? constants.O_RDWR : constants.O_RDONLY;
          if (oflags & OFLAGS_CREAT) flags |= constants.O_CREAT;
          if (oflags & OFLAGS_EXCL) flags |= constants.O_EXCL;
          if (oflags & OFLAGS_TRUNC) flags |= constants.O_TRUNC;
          if (append) flags |= constants.O_APPEND;
          try {
            this.fds.set(fd, { kind: "file", host, fd: openSync(host, flags), pos: 0, append });
          } catch (error) {
            return errnoOf(error);
          }
        }
        this.view().setUint32(out, fd, true);
        return OK;
      },
      fd_close: (fd: number) => {
        const e = this.fds.get(fd);
        if (!e) return EBADF;
        if (e.kind === "file") closeSync(e.fd);
        this.fds.delete(fd);
        return OK;
      },
      fd_read: (fd: number, iovs: number, count: number, nread: number) => {
        const e = this.fds.get(fd);
        if (!e) return EBADF;
        let total = 0;
        for (const [ptr, len] of this.iovecs(iovs, count)) {
          const n = this.read(e, this.bytes(ptr, len), null);
          if (n < 0) return EBADF;
          total += n;
          if (e.kind === "file") e.pos += n;
          if (n < len) break;
        }
        this.view().setUint32(nread, total, true);
        return OK;
      },
      fd_pread: (fd: number, iovs: number, count: number, offset: bigint, nread: number) => {
        const e = this.fds.get(fd);
        if (!e || e.kind !== "file") return EBADF;
        let total = 0;
        let at = Number(offset);
        for (const [ptr, len] of this.iovecs(iovs, count)) {
          const n = this.read(e, this.bytes(ptr, len), at);
          total += n;
          at += n;
          if (n < len) break;
        }
        this.view().setUint32(nread, total, true);
        return OK;
      },
      fd_write: (fd: number, iovs: number, count: number, nwritten: number) => {
        const e = this.fds.get(fd);
        if (!e || (e.kind !== "out" && e.kind !== "file")) return EBADF;
        let total = 0;
        for (const [ptr, len] of this.iovecs(iovs, count)) {
          const data = this.bytes(ptr, len);
          try {
            writeAll(e.fd, data, e.kind === "file" && !e.append ? e.pos : null);
          } catch {
            return EIO;
          }
          if (e.kind === "file") e.pos += len;
          total += len;
        }
        this.view().setUint32(nwritten, total, true);
        return OK;
      },
      fd_seek: (fd: number, offset: bigint, whence: number, out: number) => {
        const e = this.fds.get(fd);
        if (!e || e.kind !== "file") return EBADF;
        const base = whence === WHENCE_SET ? 0 : whence === WHENCE_CUR ? e.pos : whence === WHENCE_END ? fstatSync(e.fd).size : -1;
        if (base < 0) return EINVAL;
        const pos = base + Number(offset);
        if (pos < 0) return EINVAL;
        e.pos = pos;
        this.view().setBigUint64(out, BigInt(pos), true);
        return OK;
      },
      fd_tell: (fd: number, out: number) => {
        const e = this.fds.get(fd);
        if (!e || e.kind !== "file") return EBADF;
        this.view().setBigUint64(out, BigInt(e.pos), true);
        return OK;
      },
      fd_sync: () => OK,
      fd_datasync: () => OK,
    };
    /* A host error (a failed stat or read) becomes EIO instead of unwinding
     * through the module; only proc_exit unwinds. */
    const guarded = (call: WasiCall): WasiCall => (...args) => {
      try {
        return call(...args);
      } catch (error) {
        if (error instanceof Exit) throw error;
        return EIO;
      }
    };
    return new Proxy(calls, { get: (target, name: string) => (target[name] ? guarded(target[name]) : () => ENOSYS) });
  }
}
