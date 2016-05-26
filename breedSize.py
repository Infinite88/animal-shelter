from lxml import html
import requests

def smallDog():
    page = requests.get('http://www.smalldogplace.com/small-dog-breed-list.html')
    tree = html.fromstring(page.content)

    smallDog = tree.xpath('//span[@class="Caption CaptionCenter"]/text()')
    return smallDog

def medDog():
    page = requests.get('http://dogtime.com/dog-breeds/characteristics/medium')
    tree = html.fromstring(page.content)

    medDog = tree.xpath('//span[@class="post-title"]/text()')
    return medDog

def largeDog():
    page = requests.get('http://dogtime.com/dog-breeds/characteristics/size')
    tree = html.fromstring(page.content)

    largeDog = tree.xpath('//span[@class="post-title"]/text()')
    return largeDog