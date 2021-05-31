# -*- coding: utf-8 -*-
"""
Scraper created by OHyic
YouTube: https://www.youtube.com/watch?v=QZn_ZxpsIw4
Github: https://github.com/ohyicong/Google-Image-Scraper


Need to download Chrome Webdriver
https://chromedriver.chromium.org/downloads

and put it in webdriver folder (replace current webdriver with the new one you'll download)
                                
This version scrapes images for all dogs in the given list
"""

#Import libraries
from GoogleImageScrapper import GoogleImageScraper
import os
import pandas as pd

dogs = pd.read_csv("recommendation_dog_list.csv") # import list of dogs

# loop through each dog name in the list
for idx in range(dogs.shape[0]):
    
    #Define file path    
    webdriver_path = os.path.normpath(os.getcwd()+"/webdriver/chromedriver")
    
    image_path = os.path.normpath(os.getcwd()+"/dog_photos/" + dogs["id"][idx]) # might need to adjust according to your folder, create folder in your computer
    

    search_keys= [dogs["name"][idx] + " dog"]
    
    #Parameters
    number_of_images = 3
    headless = False
    min_resolution=(0,0)
    max_resolution=(100000,100000) 
    
    #Scrape dog images
    for search_key in search_keys:
        image_scrapper = GoogleImageScraper(webdriver_path,image_path,search_key,number_of_images,headless,min_resolution,max_resolution)
        image_urls = image_scrapper.find_image_urls()
        image_scrapper.save_images(image_urls)
    
