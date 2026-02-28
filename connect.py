import os
import sys
import logging
import gradio as gr

# RVCの内部モジュールを読み込めるようにパスを通す
now_dir = os.getcwd()
sys.path.append(now_dir)

from configs.config import Config
from infer.modules.vc.modules import VC

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. 初期設定（ConfigとVCクラスの準備）
config = Config()
vc = VC(config)

# --- 4つの主要機能の実装 ---

# A. モデル一覧取得 (infer_refresh)
def infer_refresh():
    weight_root = os.getenv("weight_root", "weights")
    names = [f for f in os.listdir(weight_root) if f.endswith(".pth")]
    return gr.Dropdown.update(choices=sorted(names))

# B. モデル切り替え (infer_change_voice)
def infer_change_voice(sid):
    # modules.py の get_vc を呼び出す
    # ※第2、第3引数は protection 設定（デフォルト値を使用）
    return vc.get_vc(sid, 0.5, 0.33)

# C. 音声変換 (infer_convert)
def infer_convert(*args):
    # modules.py の vc_single をそのまま実行
    # ねおんちゃんが追加してくれた Base64 処理が含まれた状態で返ってくるよ
    return vc.vc_single(*args)

# D. ロード状態確認 (infer_loaded_voice)
def infer_loaded_voice():
    # VCクラス内に保存されている現在のモデルIDを返す
    return getattr(vc, "loaded_model_id", "No model loaded")

# --- APIのエンドポイント定義 ---

with gr.Blocks() as app:
    # UI要素は最小限（APIとして叩くための窓口だけ作る）
    sid = gr.Dropdown(label="Model List")

    # 1. リフレッシュ
    but_refresh = gr.Button("Refresh")
    but_refresh.click(fn=infer_refresh, inputs=[], outputs=[sid], api_name="infer_refresh")

    # 2. モデル切り替え
    sid.change(fn=infer_change_voice, inputs=[sid], outputs=[], api_name="infer_change_voice")

    # 3. 変換（引数が多いので、本来は infer-web.py の引数リストと合わせる）
    # ここでは api_name="infer_convert" として定義
    dummy_output = [gr.Textbox(), gr.Audio(), gr.JSON()]
    btn_convert = gr.Button("Convert") # 実際は外部から叩かれる
    btn_convert.click(fn=vc.vc_single, inputs=[sid], outputs=dummy_output, api_name="infer_convert")

    # 4. 状態確認
    check_status = gr.Button("Check Status")
    check_status.click(fn=infer_loaded_voice, inputs=[], outputs=[gr.Textbox()], api_name="infer_loaded_voice")

# Colabで切断されないよう、queueを有効にして起動
app.queue().launch(share=True, server_name="0.0.0.0")