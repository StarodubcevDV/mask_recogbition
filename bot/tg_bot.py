import os
from io import BytesIO

import telebot
import cv2
from dotenv import load_dotenv

from detect import detect

load_dotenv()

token = os.getenv('TG_TOKEN')
bot = telebot.TeleBot(token)


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Здравствуйте! Пришлите ваше фото!')


@bot.message_handler(content_types=['photo'])
def process_img(message):
    file_id = message.photo[-1].file_id
    file_inf = bot.get_file(file_id)
    file = bot.download_file(file_inf.file_path)
    with open("image.jpg", "wb") as f:
        f.write(file)
    img = cv2.imread("image.jpg")
    bot.send_message(message.chat.id, "Начинаю поиск нарушителя!")
    detect(img)
    img_res = BytesIO(open("./res.jpg", "rb").read())
    bot.send_photo(message.chat.id, img_res)


if __name__ == '__main__':
    bot.polling(none_stop=True)
