import os
import sys
import logging
import gradio as gr

# 1. Google Driveをマウント
from google.colab import drive
drive.mount('/content/drive')

# 2. 環境変数の設定 (.env の代わり)
os.environ["weight_root"] = "/content/drive/MyDrive/RVC_Models/weights"
os.environ["index_root"] = "/content/drive/MyDrive/RVC_Models/logs"
os.environ["rmvpe_root"] = "assets/rmvpe"

# パスを通す
now_dir = os.getcwd()
sys.path.append(now_dir)

from configs.config import Config
from infer.modules.vc.modules import VC

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 3. 初期設定
config = Config()
vc = VC(config)

# --- 4つの主要機能 (API) ---

# A. モデル一覧取得 (infer_refresh)
def infer_refresh():
    root = os.getenv("weight_root")
    names = [f for f in os.listdir(root) if f.endswith(".pth")]
    return gr.Dropdown.update(choices=sorted(names))

# B. モデル切り替え (infer_change_voice)
def infer_change_voice(sid):
    # modules.py の get_vc を実行
    return vc.get_vc(sid, 0.5, 0.33) # ※第2、第3引数は protection 設定（デフォルト値を使用）

def infer_convert(*args):
    # modules.py の vc_single を実行
    return vc.vc_single(*args)

# D. ロード状態確認 (infer_loaded_voice)
def infer_loaded_voice():
    # VCクラス内に保存されている現在のモデルIDを返す
    return getattr(vc, "loaded_model_id", "No model loaded")

# --- APIのエンドポイント定義 ---

with gr.Blocks() as app:
    # 最小限のUI要素
    sid = gr.Dropdown(label="Model List")

    # 1. リフレッシュ
    btn_ref = gr.Button("Refresh")
    btn_ref.click(fn=infer_refresh, inputs=[], outputs=[sid], api_name="infer_refresh")

    # 2. モデル切り替え
    sid.change(fn=infer_change_voice, inputs=[sid], outputs=[], api_name="infer_change_voice")

    # 3. 変換（出力は修正版に合わせて [テキスト, 音声, JSON]）
    btn_cnv = gr.Button("Convert")
    btn_cnv.click(fn=vc.vc_single, inputs=[sid], outputs=[gr.Textbox(), gr.Audio(), gr.JSON()], api_name="infer_convert")

    # 4. 状態確認
    btn_sts = gr.Button("Status")
    btn_sts.click(fn=infer_loaded_voice, inputs=[], outputs=[gr.Textbox()], api_name="infer_loaded_voice")

# 起動
app.queue().launch(share=True, server_name="0.0.0.0")
