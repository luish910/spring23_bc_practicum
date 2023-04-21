# Spring 2023 Budget Collector Practicum - Group 6

This is a repository for the back end of our project, for the front-end please use this link: https://github.com/bwolfram1/Spring2023-BC-Prac3

## Modelling:

- Input: WikiArt5Colors.csv
- Create folders: images, resized-images, display-images, model-output, saved-models
- Run import-images.py
- Run create-models.py


## API:
server: https://spring23-bc-group6.onrender.com

/raw_values
Returns full dataset

/region
Parameters:
  year_from
  year_to
Returns counts by region

/movement
Parameters:
  year_from
  year_to
  region
Returns colors by movement and region


/paintings
Parameters:
  year_from
  year_to
  region
  movement
  img_folder: this is the image folder name on your project, it used to return an image path
Returns painting metadata within movement and region
