from PyInstaller.__main__ import run
if __name__ == '__main__':
    opt = ['-F','--noupx','--clean',
           'seg_predict_task1.py']
    run(opt)