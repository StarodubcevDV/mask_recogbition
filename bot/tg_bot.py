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
    bot.send_message(message.chat.id, 'Здравствуйте! Пришлите ваше фото!\nДокументы не принимаются!😀')


@bot.message_handler(content_types=["sticker"])
def start_message(message):
    bot.send_message(message.chat.id, '😀')


@bot.message_handler(content_types=['text'])
def start_message(message):
    if message.text == '😀':
        bot.send_message(message.chat.id, '😀')
    else:
        bot.send_message(message.chat.id, 'Извините, меня не научили вас понимать.. 😞 \nЛучше пришлите мне ваше фото!')


@bot.message_handler(content_types=['audio', 'video', 'video_note', 'document'])
def start_message(message):
    bot.send_message(message.chat.id, 'Такое я не могу распознать.. 😞')


@bot.message_handler(content_types=['photo'])
def process_img(message):
    file_id = message.photo[-1].file_id
    file_inf = bot.get_file(file_id)
    file = bot.download_file(file_inf.file_path)
    with open("image.jpg", "wb") as f:
        f.write(file)
    img = cv2.imread("image.jpg")
    bot.send_message(message.chat.id, "Начинаю поиск нарушителя!")
    det_info = detect(img)
    img_res = BytesIO(open("res.jpg", "rb").read())
    bot.send_photo(message.chat.id, img_res)
    if len(det_info.keys()) == 2:
        bot.send_message(message.chat.id, f'В масках: {det_info["mask"]} \nБез масок: {det_info["no_mask"]}')
    elif len(det_info.keys()) == 1:
        if list(det_info.keys())[0] == 'mask':
            bot.send_message(message.chat.id, f'В масках: {det_info["mask"]} \nБез масок: 0')
        elif list(det_info.keys())[0] == 'no_mask':
            bot.send_message(message.chat.id, f'В масках: 0 \nБез масок: {det_info["no_mask"]}')
    elif len(det_info.keys()) == 0:
        bot.send_message(message.chat.id, 'Я не смог никого найти 😞')
