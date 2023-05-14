import os
from src.app_gradio import make_frontend

pth = os.getcwd() + "/src/gifsplanation"
if not os.path.exists(pth):
    os.system("git clone https://github.com/mlmed/gifsplanation " + pth)

if __name__ == "__main__":
    frontend = make_frontend()
    frontend.launch(share=True)
