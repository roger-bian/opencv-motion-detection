import glob
import cv2
import numpy as np
import time


file_names = glob.glob('2024*_orig.mp4')

for file_name in file_names:
    cap = cv2.VideoCapture(file_name)

    red = (0, 0, 255) # 枠線の色
    before = None # 前回の画像を保存する変数
    fps = 5 #動画のFPSを取得

    writer = cv2.VideoWriter('output_' + file_name, cv2.VideoWriter_fourcc(*'MP4V'), fps, (1920, 960*2))

    frame = None

    while True:
        # 画像を取得
        ret, frame = cap.read()
        # 再生が終了したらループを抜ける
        if ret:
            frame = cv2.resize(frame, (1920, 960))
            
            # 白黒画像に変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if before is None:
                before = gray.astype("float")
                continue
            
            #現在のフレームと移動平均との差を計算
            cv2.accumulateWeighted(gray, before, 0.8)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(before))
            #frameDeltaの画像を２値化
            _, thresh = cv2.threshold(frameDelta, 10, 255, cv2.THRESH_BINARY)
            #postprocessing
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            thresh = cv2.dilate(thresh, np.ones((11, 11), np.uint8))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
            # clean top bar due to clock
            thresh[:50, :] = 0
            
            #輪郭のデータを得る
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 差分があった点を画面に描く
            cnt_temp = np.zeros_like(thresh)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w*h < 1200:
                    continue # 小さな変更点は無視
                cv2.rectangle(cnt_temp, (x-10, y-10), (x+w+10, y+h+10), (255), -1)
            cnt_temp = cv2.morphologyEx(cnt_temp, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
                
            # create new canvas for cleaning overlapping detection rectangles
            contours, _ = cv2.findContours(cnt_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)

            #ウィンドウでの再生速度を元動画と合わせる
            time.sleep(1/fps)
            
            final = np.vstack([
                cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
                frame
            ])
            # save video
            writer.write(final)
            # ウィンドウで表示
            cv2.imshow('target_frame', final)
            
            # Enterキーが押されたらループを抜ける
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows() # ウィンドウを破棄
