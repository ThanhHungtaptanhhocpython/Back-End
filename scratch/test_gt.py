import sys
import traceback
from deep_translator import GoogleTranslator

try:
    gt = GoogleTranslator(source='vi', target='en')
    text = "Đoạn clip là cảnh thu hoạch dứa ở miền Tây: một bà cụ ngồi bên giỏ dứa trò chuyện với cô gái mặc áo hồng quàng khăn rằn"
    res = gt.translate(text)
    print("SUCCESS:", repr(res))
except Exception as e:
    print("ERROR:", type(e), e)
    traceback.print_exc()
