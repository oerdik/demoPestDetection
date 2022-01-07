import cv2
from tracker import * #tracker dosyasını çağırdık

# tracker objesini oluşturduk
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("pestiess.mp4")

# sabit kameradan nesne algılama
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # Çalışma alanı belirleme
    roi = frame[500: 1200,500: 1200]

    # 1. Nesne Algılama
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY) #Mask gölgeleri algılamasın
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = [] #her nesene için bir dizi
    for cnt in contours:
        # çalışma alanı tanıma hassasiyeti 
        area = cv2.contourArea(cnt)
        if area > 50:
            #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2) Nesneleri kenarıyla belirtme
            x, y, w, h = cv2.boundingRect(cnt)


            detections.append([x, y, w, h]) #diziye yazdırma

    # 2. Nesne İzleme
    boxes_ids = tracker.update(detections)#her nesneye id verme
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)#nesneleri dikdörtgen içine alma
        if id > 5:#tespit edilen nesne sayısı 5i geçince ilaçlama bildirimi 
            print("İlaçlama Zamanı")

    cv2.imshow("roi", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
