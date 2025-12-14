car_x1, car_x2, car_y1, car_y2 = frame["car_coords"]
        
        # Рисуем прямоугольник для автомобиля
        cv2.rectangle(img_copy, 
                    (car_x1, car_y1), 
                    (car_x2, car_y2), 
                    CAR_COLOR, 2)
        
        # Извлекаем координаты номерного знака
        lp_x1, lp_x2, lp_y1, lp_y2 = car["lp_coords"]
        
        # Рисуем прямоугольник для номерного знака (более толстый)
        cv2.rectangle(img_copy, 
                    (lp_x1, lp_y1), 
                    (lp_x2, lp_y2), 
                    LP_COLOR, 3)
        
        # Получаем текст номера
        lp_text = car.get("lp_text", "")
        
        # Добавляем текст над bounding box'ом автомобиля
        if lp_text:
            # Фон для текста номера
            text_size = cv2.getTextSize(lp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Позиционируем текст над bounding box'ом номера
            text_x = lp_x1
            text_y = lp_y1 - 10 if lp_y1 - 10 > 10 else lp_y2 + 25
            
            # Рисуем фон для текста
            cv2.rectangle(img_copy,
                        (text_x, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 10, text_y + 5),
                        TEXT_BG_COLOR,
                        -1)
            
            # Рисуем текст номера
            cv2.putText(img_copy,
                    lp_text,
                    (text_x + 5, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    TEXT_COLOR,
                    2)
        
        # Добавляем метку "Car" над bounding box'ом автомобиля
        cv2.putText(img_copy,
                "Car",
                (car_x1, car_y1 - 10 if car_y1 - 10 > 10 else car_y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                CAR_COLOR,
                2)