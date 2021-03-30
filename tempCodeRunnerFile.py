    # for (x,y,w,h) in faces:
    #     if (w < 150 or h < 150 and len(faces) == 1):
    #         continue
        
    #     frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    #     roi_gray = gray[y:y+h, x:x+w]
    #     roi_color = frame[y:y+h, x:x+w]
    #     gray_img = gray[x:x+w, y:y+h]
    #     if ( len(gray_img) == 0):
    #         continue

    #     gray_img = cv2.resize(gray_img,dsize=(48,48))
    #     gray_img = gray_img.reshape((1,48,48,1))
    #     #print(gray_img.shape)
    #     predictions = model.predict( gray_img )
    #     pred = [np.argmax(i) for i in predictions]
    #     #print(predictions)
    #     cv2.putText(frame, emotion[pred[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
