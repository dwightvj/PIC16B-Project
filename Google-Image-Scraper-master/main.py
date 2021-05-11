# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:02:06 2020

@author: OHyic
https://www.youtube.com/watch?v=QZn_ZxpsIw4
https://github.com/ohyicong/Google-Image-Scraper

Need to download Chrome Webdriver
https://chromedriver.chromium.org/downloads

and put it in webdriver folder (replace current webdriver with the new one you'll download)
"""

#Import libraries
from GoogleImageScrapper import GoogleImageScraper
import os

#Define file path
webdriver_path = os.path.normpath(os.getcwd()+"/webdriver/chromedriver")
image_path = os.path.normpath(os.getcwd()+"/pitbulls") # might need to adjust according to your folder

#Add new search key into array ["cat","t-shirt","apple","orange","pear","fish"]
search_keys= ["pitbulls"]

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