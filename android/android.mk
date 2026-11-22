# The Android app (docs/ANDROID.md): `make apk` in the top Makefile calls this
# file with LIB (the runtime's C sources), BUNDLE_MODEL (a self-contained
# .smoe) and MODEL_NAME. No Gradle: the NDK's clang builds libsmolmoe.so,
# javac and d8 build the small MainActivity, aapt packs, zipalign aligns and
# apksigner signs. Needs ANDROID_HOME (the SDK, with an NDK, build-tools and a
# platform) and a JDK.
#   make apk                       dist/smolmoe-<model>.apk (arm64-v8a, phones)
#   make apk APK_ABI=x86_64        dist/smolmoe-<model>-x86_64.apk, for the emulator on a PC
#   APK_KEYSTORE=... APK_KS_PASS=... APK_KEY_ALIAS=...   sign with your own key

ifeq ($(ANDROID_HOME),)
  $(error ANDROID_HOME is not set: point it at the Android SDK (with ndk/, build-tools/ and platforms/))
endif
SDK := $(subst \,/,$(ANDROID_HOME))
latest = $(lastword $(sort $(wildcard $(1))))
NDK ?= $(or $(subst \,/,$(ANDROID_NDK_HOME)),$(call latest,$(SDK)/ndk/*))
BUILD_TOOLS ?= $(call latest,$(SDK)/build-tools/*)
ANDROID_JAR ?= $(call latest,$(SDK)/platforms/android-*/android.jar)
ifeq ($(NDK),)
  $(error no NDK under $(SDK)/ndk: install one with sdkmanager "ndk;<version>")
endif
ifeq ($(BUILD_TOOLS),)
  $(error no build-tools under $(SDK)/build-tools)
endif
ifeq ($(ANDROID_JAR),)
  $(error no platform under $(SDK)/platforms)
endif

ifeq ($(OS),Windows_NT)
  HOST_TAG := windows-x86_64
  HOST_EXE := .exe
  BAT      := .bat
else
  HOST_TAG := $(if $(filter Darwin,$(shell uname -s)),darwin-x86_64,linux-x86_64)
  HOST_EXE :=
  BAT      :=
endif
TOOLCHAIN := $(NDK)/toolchains/llvm/prebuilt/$(HOST_TAG)/bin
API       := 29
APK_ABI   ?= arm64-v8a
ifeq ($(APK_ABI),arm64-v8a)
  TRIPLE := aarch64-linux-android
  SUFFIX :=
else ifeq ($(APK_ABI),x86_64)
  TRIPLE := x86_64-linux-android
  SUFFIX := -x86_64
else
  $(error APK_ABI must be arm64-v8a or x86_64)
endif
TARGET    := --target=$(TRIPLE)$(API)
ACC       := $(TOOLCHAIN)/clang$(HOST_EXE) $(TARGET)
ACXX      := $(TOOLCHAIN)/clang++$(HOST_EXE) $(TARGET)
AAPT      := $(BUILD_TOOLS)/aapt$(HOST_EXE)
ZIPALIGN  := $(BUILD_TOOLS)/zipalign$(HOST_EXE)
APKSIGNER := $(BUILD_TOOLS)/apksigner$(BAT)
D8        := $(BUILD_TOOLS)/d8$(BAT)
JAVAC     ?= javac
GLUE      := $(NDK)/sources/android/native_app_glue

APK_KEYSTORE  ?= $(HOME)/.android/debug.keystore
APK_KS_PASS   ?= android
APK_KEY_ALIAS ?= androiddebugkey
VERSION       := $(shell sed -n 's/^\#define SMOLMOE_VERSION "\(.*\)"/\1/p' src/util.h)

OUT    := build/android/$(APK_ABI)
ASSETS := $(OUT)/assets-$(MODEL_NAME)
DIST   ?= dist
APK    := $(DIST)/smolmoe-$(MODEL_NAME)$(SUFFIX).apk
IMGUI  := third_party/imgui

# The runtime's own warnings and determinism flags (top Makefile), with -fPIC.
STD     := -std=c99 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wstrict-prototypes -Wconversion
FPFLAGS := -ffp-contract=off -fno-fast-math
OPT     := -O3 -fPIC -pthread
CXXSTD  := -std=c++17

C_OBJ     := $(addprefix $(OUT)/obj/,$(addsuffix .o,$(LIB) desktop))
GUI_OBJ   := $(addprefix $(OUT)/gui/,platform_android.o session.o view.o)
IMGUI_OBJ := $(addprefix $(OUT)/imgui/,$(addsuffix .o,imgui imgui_draw imgui_tables imgui_widgets misc/cpp/imgui_stdlib \
	backends/imgui_impl_android backends/imgui_impl_opengl3))
GLUE_OBJ  := $(OUT)/glue/android_native_app_glue.o
SO        := $(OUT)/apk/lib/$(APK_ABI)/libsmolmoe.so

.PHONY: apk
apk: $(APK)

$(OUT)/obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(ACC) $(OPT) $(STD) $(FPFLAGS) -MMD -MP -c $< -o $@

$(OUT)/gui/%.o: gui/%.cpp
	@mkdir -p $(dir $@)
	$(ACXX) $(OPT) $(CXXSTD) -Wall -Wextra -Wpedantic -Werror -Isrc -I$(IMGUI) -I$(IMGUI)/backends -I$(GLUE) -MMD -MP \
		-c $< -o $@

$(OUT)/imgui/%.o: $(IMGUI)/%.cpp
	@mkdir -p $(dir $@)
	$(ACXX) $(OPT) $(CXXSTD) -I$(IMGUI) -DIMGUI_IMPL_OPENGL_ES3 -MMD -MP -c $< -o $@

$(GLUE_OBJ): $(GLUE)/android_native_app_glue.c
	@mkdir -p $(dir $@)
	$(ACC) $(OPT) -c $< -o $@

# ANativeActivity_onCreate lives in the glue and is only referenced by the
# system, so it is kept by name. 16 KiB pages: Android 15 devices may use them.
$(SO): $(C_OBJ) $(GUI_OBJ) $(IMGUI_OBJ) $(GLUE_OBJ)
	@mkdir -p $(dir $@)
	$(ACXX) -shared -o $@ $^ -static-libstdc++ -Wl,-s -Wl,--no-undefined -Wl,-z,max-page-size=16384 \
		-u ANativeActivity_onCreate -landroid -llog -lEGL -lGLESv3 -lm

$(OUT)/apk/classes.dex: android/java/io/smolmoe/app/MainActivity.java
	@rm -rf $(OUT)/classes && mkdir -p $(OUT)/classes $(OUT)/apk
	$(JAVAC) --release 11 -Xlint:-options -classpath "$(ANDROID_JAR)" -d $(OUT)/classes $<
	$(D8) --release --min-api $(API) --lib "$(ANDROID_JAR)" --output $(OUT)/apk $(OUT)/classes/io/smolmoe/app/*.class

$(ASSETS)/model.smoe: $(BUNDLE_MODEL)
	@mkdir -p $(dir $@)
	cp $< $@
	printf '%s\n' '$(MODEL_NAME)' > $(ASSETS)/model-name.txt

# aapt stores everything uncompressed (-0 ''): the library and the model are
# mapped straight from the APK. zipalign then puts the library on a 16 KiB page.
$(APK): $(SO) $(OUT)/apk/classes.dex $(ASSETS)/model.smoe android/AndroidManifest.xml android/res/values/styles.xml android/res/values-night/styles.xml
	@mkdir -p $(DIST)
	sed 's/@VERSION@/$(VERSION)/' android/AndroidManifest.xml > $(OUT)/AndroidManifest.xml
	rm -f $(OUT)/unaligned.apk $(OUT)/aligned.apk
	$(AAPT) package -f -0 '' -M $(OUT)/AndroidManifest.xml -S android/res -I "$(ANDROID_JAR)" -A $(ASSETS) \
		-F $(OUT)/unaligned.apk $(OUT)/apk
	$(ZIPALIGN) -f -P 16 4 $(OUT)/unaligned.apk $(OUT)/aligned.apk
	$(APKSIGNER) sign --ks "$(APK_KEYSTORE)" --ks-pass pass:$(APK_KS_PASS) --ks-key-alias $(APK_KEY_ALIAS) \
		--out $@ $(OUT)/aligned.apk
	$(APKSIGNER) verify $@
	@echo "APK pronto: $@"

-include $(C_OBJ:.o=.d) $(GUI_OBJ:.o=.d) $(IMGUI_OBJ:.o=.d)
