{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec3a24fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from telegram import Bot\n",
    "from telegram.utils.request import Request\n",
    "import os\n",
    "\n",
    "# Inserisci il tuo token del bot Telegram\n",
    "TELEGRAM_BOT_TOKEN = '7405803640:AAFMdlQPg4oa7M94WLdNcyof8fnngyI2Qns'\n",
    "# Inserisci l'ID della chat Telegram a cui inviare la notifica\n",
    "CHAT_ID = '6484742622'\n",
    "\n",
    "def send_telegram_message(message):\n",
    "    request = Request(con_pool_size=8)\n",
    "    bot = Bot(token=TELEGRAM_BOT_TOKEN, request=request)\n",
    "    bot.send_message(chat_id=CHAT_ID, text=message)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    send_telegram_message(\"Ho fattoooo :) \")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.16"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
