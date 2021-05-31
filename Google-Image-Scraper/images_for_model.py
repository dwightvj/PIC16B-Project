# -*- coding: utf-8 -*-
"""
Scraper created by OHyic
YouTube: https://www.youtube.com/watch?v=QZn_ZxpsIw4
Github: https://github.com/ohyicong/Google-Image-Scraper


Need to download Chrome Webdriver
https://chromedriver.chromium.org/downloads

and put it in webdriver folder (replace current webdriver with the new one you'll download)
                                
This version of the code scrapes many images. (good for collecting many dog images for our model)
"""

#Import libraries
from GoogleImageScrapper import GoogleImageScraper
import os

#Define file path
webdriver_path = os.path.normpath(os.getcwd()+"/webdriver/chromedriver")
image_path = os.path.normpath(os.getcwd()+"/australian_shepherd") # might need to adjust according to your folder

#Add new search key into array
search_keys = ["Australian Shepherd dog"]

#Parameters
number_of_images = 200
headless = False
min_resolution=(0,0)
max_resolution=(2000,2000) 

#Main program
for search_key in search_keys:
    image_scrapper = GoogleImageScraper(webdriver_path,image_path,search_key,number_of_images,headless,min_resolution,max_resolution)
    image_urls = image_scrapper.find_image_urls()
    image_scrapper.save_images(image_urls)