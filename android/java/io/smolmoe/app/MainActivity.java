// The Android app: a NativeActivity whose native side (gui/platform_android.cpp)
// draws the chat. Java only does what the native side cannot do well: the
// system's text box, with the keyboard's accents, autocorrect and paste, and
// the clipboard.
package io.smolmoe.app;

import android.app.AlertDialog;
import android.app.NativeActivity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.text.InputType;
import android.view.WindowManager;
import android.widget.EditText;
import android.widget.Toast;

public class MainActivity extends NativeActivity {
    static {
        // NativeActivity loads the library on its own; loading it here as well
        // binds onText, which the native side defines.
        System.loadLibrary("smolmoe");
    }

    private static final int PADDING_DP = 20;

    // The text box closed: send is true for "Enviar", false keeps the draft.
    static native void onText(String text, boolean send);

    // Called from the native thread with the current draft.
    public void editText(final String draft) {
        runOnUiThread(() -> {
            final EditText field = new EditText(this);
            field.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                    | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES | InputType.TYPE_TEXT_FLAG_AUTO_CORRECT);
            field.setMinLines(2);
            field.setMaxLines(8);
            field.setText(draft);
            field.setSelection(field.getText().length());
            field.setHint("Sua mensagem");
            final int pad = Math.round(PADDING_DP * getResources().getDisplayMetrics().density);
            field.setPadding(pad, pad, pad, pad);
            AlertDialog dialog = new AlertDialog.Builder(this)
                    .setView(field)
                    .setPositiveButton("Enviar", (d, which) -> onText(field.getText().toString(), true))
                    .setNegativeButton("Fechar", (d, which) -> onText(field.getText().toString(), false))
                    .setOnCancelListener(d -> onText(field.getText().toString(), false))
                    .create();
            dialog.getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_STATE_ALWAYS_VISIBLE);
            dialog.show();
            field.requestFocus();
        });
    }

    // Called from the native thread (Dear ImGui's clipboard).
    public void copyText(final String text) {
        runOnUiThread(() -> {
            ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            clipboard.setPrimaryClip(ClipData.newPlainText("smolmoe", text));
            Toast.makeText(this, "Copiado", Toast.LENGTH_SHORT).show();
        });
    }
}
