# spring23_bc_practicum

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
