import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import davies_bouldin_score
import joblib
import cv2


def rgb_to_lab2(rgb):
    rgb_pixel = np.uint8(rgb)
    rgb_pixel = rgb_pixel.reshape(1,1,3)
    return cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2Lab)

def lab_to_rgb2(lab):
    lab_pixel = np.uint8(lab)
    lab_pixel = lab_pixel.reshape(1,1,3)
    return cv2.cvtColor(lab_pixel, cv2.COLOR_Lab2RGB)


#bakground removal, for a given image check the pixels on the the edges (8 pixels).
def remove_bc_and_reshape (image, rgb_factor, scaled = False):
    result = False
    bc = None
    h,w,c = image.shape
    points = [
            (0, 0),(0, w - 1),(h-1,0),(h - 1, w - 1) #corners
            , (0, int(w/2)),  (h-1, int(w/2)) , (int(h/2), 0) ,(int(h/2), w-1)    #middel pixels on edges
            ]
    rounded_pixels = []
    for p in points:
        if scaled:
            rounded_pixels.append(np.round(image[p]*255/rgb_factor)*rgb_factor)
        else:    
            rounded_pixels.append(np.round(image[p]/rgb_factor)*rgb_factor)
    unq, cnt = np.unique(rounded_pixels,return_counts=True, axis=0)
    image = image.reshape(-1,3) #reshape to pixel array
    original_image = image #save original
    if cnt.max() >= 3:
        result = True
        bc = unq[cnt == cnt.max()][0]
        if scaled:
            image = image[np.where((np.any(image*255 <= bc - rgb_factor,axis=1)) | (np.any(image*255 > bc + rgb_factor,axis=1) ))]
        else: 
            image = image[np.where((np.any(image <= bc - rgb_factor,axis=1)) | (np.any(image > bc + rgb_factor,axis=1) ))]
    if image.shape[0] == 0:
       image = original_image  #at least must keep some pixels        
    return image, result, bc


#RGB TO HEX FUNCTION
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

import matplotlib.colors as mcolors
def rgb_to_hex(rgb):
    r,g,b = rgb
   
    return mcolors.to_hex((r/255,g/255,b/255))

#load data
df = pd.read_csv("WikiArt5Colors.csv")


#FITTING
#FIT MODELS by USING RGB, CIELAB, CIELAB 2D 
#EVALUATE K by using ELBOW METHOD
#EVALUATE METHOD PERFORMANCE USING DB because it is bounded
#config
k_values = [4,5,6,7,8,9,10,11,12,13,14,15] #since we are performing whole movement we can be less granular
methods = ['rgb','lab_3d','lab_2d']
folder = 'resized-images' #resized 100*100 images
model_path = 'saved-models'
results = []
#run it for all model combinations 
regions = df['NatRegion'].unique()
df_luminosity_all = pd.DataFrame()

for reg in regions:
    #create a subset
    df_region = df[df['NatRegion'] == reg]
    movements = df_region['Cat2'].unique()
    for mov in movements:
        print('region: ',reg,' - movement:',mov)
        #create a subset
        df_region_movement = df_region[df_region['Cat2']== mov]['ID']
        pixel_matrix = []
        df_luminosity = []
        
        #load movement and region images
        images = []
        for id in df_region_movement:
            full_path = os.path.join(folder, str(id) + ".jpg")
            images.append(plt.imread(full_path)) 

        #pre-process data for each method
        pmatrix_rgb= None
        pmatrix_lab_3d = None
        pmatrix_lab_2d = None

        #rgb
        for img in images:
            img = img.reshape(-1,3)        
            if pmatrix_rgb is None:
                pmatrix_rgb = img
            else:
                pmatrix_rgb = np.concatenate((pmatrix_rgb, img))    
        #lab
        pmatrix_lab_3d = np.apply_along_axis(rgb_to_lab2, 1, pmatrix_rgb)      
        pmatrix_lab_3d = pmatrix_lab_3d.reshape(-1,3)
        #lab_3d
        pmatrix_lab_2d = pmatrix_lab_3d[:,1:3]
        print("pre-processing finished")

        #FITTING
        pixel_matrix = [pmatrix_rgb,pmatrix_lab_3d,pmatrix_lab_2d]

        for i in range(3):
            #tuning: 
            wcss = [] 
            db_scores = []
            models = []
            for k in k_values:
                kmeans = KMeans(n_clusters = k
                                ,init = 'k-means++' #this is better than random! and might converge faster
                                ,n_init= 'auto'
                                ,random_state=2023
                                ,max_iter=300)
                kmeans.fit(pixel_matrix[i])
                #print(kmeans.labels_.shape)
                wcss.append(kmeans.inertia_)
                models.append(kmeans)
                db_scores.append(davies_bouldin_score(pixel_matrix[i],kmeans.labels_))
            #chose a best k based elbow method. save best k.
            kn = KneeLocator(k_values, wcss, curve='convex', direction='decreasing')
            if kn.knee == None:
                best_k = k_values[0] #default to 5-color palette
                kmeans_model = models[0]   
                best_wcss = wcss[0]
                db_score = db_scores[0]
            else:
                best_k = kn.knee
                idx = k_values.index(best_k)
                kmeans_model = models[idx]
                best_wcss = wcss[idx]
                db_score = db_scores[idx]

            #save the model to a folder. save  scores  
            model_id =  methods[i]+"_" +reg.replace(" ","")+"_"+mov.replace(" ","")
            filename = os.path.join(model_path, model_id+".pkl")
            joblib.dump(kmeans_model, filename)  
            results.append([reg,mov,methods[i],best_k,best_wcss,db_score,df_region_movement.shape[0]])

            #export luminosity for lab2d
            if methods[i] == 'lab_2d':
                luminosity = pmatrix_lab_3d[:,0] #extract from 3D matrix
                cluster = kmeans_model.labels_
                df_luminosity = pd.DataFrame({'luminosity': luminosity, 'cluster': cluster}, columns=['luminosity', 'cluster'])
                df_luminosity = df_luminosity.groupby('cluster').median()
                df_luminosity = df_luminosity.reset_index()
                df_luminosity['model_id'] = model_id
                df_luminosity_all = pd.concat([df_luminosity_all, df_luminosity])

#export results
result_columns = ['region','movement','method','best_k','best wcss','db_score','paintings']
df_results = pd.DataFrame(results)
df_results.columns = result_columns
df_results.to_csv('model_results_3methods.csv', index = False)
#export luminosity
df_luminosity_all.to_csv('df_luminosity_all.csv', index = False)


#Plot the palettes
#plot results for region + movement
plot_methods = ['RGB','CIELAB 3D','CIELAB 2D']
is_lab = [False,True,True]
is_2d = [False,False,True]
regions = df['NatRegion'].unique()
df_luminosity = pd.read_csv("df_luminosity_all.csv")
for reg in regions:
    #create a subset
    df_region = df[df['NatRegion'] == reg]
    movements = df_region['Cat2'].unique()
    for mov in movements:
        plt.figure(figsize=(32,8)) 
        plt.suptitle('Region: '+reg+' - Movement: '+mov,fontsize = 28)
        for i in range(3):
            model_id = methods[i]+'_'+reg.replace(" ","")+"_"+mov.replace(" ","")
            filename = os.path.join(model_path, model_id+".pkl")
            kmeans_model = joblib.load(filename)
            counts = np.array(np.unique(kmeans_model.labels_,return_counts = True)).T

            palette =  kmeans_model.cluster_centers_
            if is_2d[i]:
                luminosity = df_luminosity[df_luminosity['model_id'] == model_id]
                median_luminosity = np.array(luminosity.groupby('cluster').median(numeric_only = True)['luminosity'])
                median_luminosity = median_luminosity.reshape(median_luminosity.shape[0],1)
                palette = np.append(median_luminosity, palette, axis=1)

            if is_lab[i]: 
                palette = np.apply_along_axis(lab_to_rgb2, 1, palette) #back_to_rgb
                palette = palette.reshape(-1,3) #required only for open CV

            palette = list(palette/255) 
            plt.subplot(1,3,i+1)  
            plt.pie(x=counts[:,1], colors = palette,autopct='%.0f%%')
            title = 'Method: ' +plot_methods[i]+ '(k='+str(kmeans_model.n_clusters)+')'
            plt.title(title, bbox={'facecolor':'0.8', 'pad':3})
plt.show()


#FIT IMAGES
#fit all paintings to the corresponding palette and extract the primary color
#initialize df to save data
df_colors = pd.DataFrame(columns=['ID','color1','color2','prop1','prop2'])
regions = df['NatRegion'].unique()
folder = 'resized-images' #original 100*100 images
rgb_factor = 15 #rounding rgb px for bg_removal
df_luminosity = pd.read_csv("df_luminosity_all.csv")

#min_saturation = 0.10 #hsv min saturation value
#max_removal = 0.40 #max % removed pixels due low saturation value
model_path = 'saved-models'
method = 'lab_2d' #selected method
for reg in regions:
    #create a subset
    df_region = df[df['NatRegion'] == reg]
    movements = df_region['Cat2'].unique()
    for mov in movements:
        #load the model
        model_id = method+'_'+reg.replace(" ","")+"_"+mov.replace(" ","")
        filename = os.path.join(model_path, model_id +".pkl")
        kmeans_model = joblib.load(filename)   

        #get centroids
        centroids =  kmeans_model.cluster_centers_ #LAB 2D
        luminosity = df_luminosity[df_luminosity['model_id'] == model_id]
        median_luminosity = np.array(luminosity.groupby('cluster').median(numeric_only = True)['luminosity'])
        median_luminosity = median_luminosity.reshape(median_luminosity.shape[0],1)
        centroids = np.append(median_luminosity, centroids, axis=1)
        centroids = np.apply_along_axis(lab_to_rgb2, 1, centroids) #back_to_rgb
        centroids = centroids.reshape(-1,3) #required only for open CV
        centroids = np.apply_along_axis(rgb_to_hex, 1, centroids) #to HEX
        
        #create a subset of the data
        df_region_movement = df_region[df_region['Cat2']== mov]['ID']
        #load  images
        images = []
        for id in df_region_movement:
            full_path = os.path.join(folder, str(id) + ".jpg")
            img = plt.imread(full_path)
            img, _ , _  = remove_bc_and_reshape(img,rgb_factor, scaled = False)   #also reshapes
            img = np.apply_along_axis(rgb_to_lab2, 1, img) #to CIELAB
            img = img.reshape(-1,3)
 
            #predict using kmeans model
            pred = kmeans_model.predict(img[:,1:3]) 
            #count pixels and sort colors
            ind, count = np.unique(pred, return_counts=True)
            proportions = count/np.sum(count)
            count_sort_ind = np.argsort(-count)
            sorted_index = ind[count_sort_ind]   

            color1 = centroids[sorted_index[0]]
            prop1 = proportions[count_sort_ind[0]]

            if len(ind) > 1:
                color2 = centroids[sorted_index[1]]
                prop2 = proportions[count_sort_ind[1]]
            else:
                color2 = ''
                prop2 = ''   
            #save data
            df_colors.loc[len(df_colors)] = [id, color1, color2, prop1, prop2]
##merge to original dataset
cols = ['ID', 'Year_split', 'Year_5', 'Artist', 'Title', 'Style2', 'Cat2',
       'NatRegion', 'Ave. art rating', 'Sentiment', 'Colorfulness', 'Image URL']
df_merge = df[cols]
df_merge = df_merge.merge(df_colors,how='inner',on='ID').fillna(0)

#save csv
df_merge.to_csv("model-output/WikiArtFull_LAB2Dcolors.csv", index = False)

# %%
#compute dominant color per region, movement and period using 5 year groups
year_group_column = 'Year_5'
df_colors_movement = pd.DataFrame(columns=['NatRegion','Cat2','Year_5','color1','color2','prop1','prop2'])
regions = df['NatRegion'].unique()
folder = 'resized-images' #resized 100*100 images
model_path = 'saved-models'
df_luminosity = pd.read_csv("df_luminosity_all.csv")
method = 'lab_2d' #selected method
rgb_factor = 15 #rounding rgb px for bg_removal

for reg in regions:
    #create a subset
    df_region = df[df['NatRegion'] == reg]
    movements = df_region['Cat2'].unique()
    for mov in movements:
        #load the model
        model_id = method+'_'+reg.replace(" ","")+"_"+mov.replace(" ","")
        filename = os.path.join(model_path, model_id +".pkl")
        kmeans_model = joblib.load(filename)   

        #get centroids
        centroids =  kmeans_model.cluster_centers_ #LAB 2D
        luminosity = df_luminosity[df_luminosity['model_id'] == model_id]
        median_luminosity = np.array(luminosity.groupby('cluster').median(numeric_only = True)['luminosity'])
        median_luminosity = median_luminosity.reshape(median_luminosity.shape[0],1)
        centroids = np.append(median_luminosity, centroids, axis=1)
        centroids = np.apply_along_axis(lab_to_rgb2, 1, centroids) #back_to_rgb
        centroids = centroids.reshape(-1,3) #required only for open CV
        centroids = np.apply_along_axis(rgb_to_hex, 1, centroids) #to HEX
        #create a subset of the data
        df_region_movement = df_region[df_region['Cat2']== mov][['ID',year_group_column]]
        year_groups = df_region_movement[year_group_column].unique()
 
        for year in year_groups:
            print(reg,' movement: ',mov, '. Year: ',year)
            #create a painting subset
            df_year = df_region_movement[df_region_movement[year_group_column]==year]['ID']

            #load movement and region and year images
            images = []
            for id in df_year:
                full_path = os.path.join(folder, str(id) + ".jpg")
                images.append(plt.imread(full_path)) 
            #pre-process data for each method
            pixel_matrix= None
            for img in images:
                img, _ , _  = remove_bc_and_reshape(img,rgb_factor, scaled = False)   #also reshapes
                #img = img.reshape(-1,3)
                if pixel_matrix is None:
                    pixel_matrix = img
                else:
                    pixel_matrix = np.concatenate((pixel_matrix, img))    
            #print(pixel_matrix.shape) 
            pixel_matrix = np.apply_along_axis(rgb_to_lab2, 1, pixel_matrix)      
            pixel_matrix = pixel_matrix.reshape(-1,3)
            pixel_matrix = pixel_matrix[:,1:3]

            #predict using kmeans model
            pred = kmeans_model.predict(pixel_matrix) 
            #count pixels and sort colors
            ind, count = np.unique(pred, return_counts=True)
            proportions = count/np.sum(count)
            count_sort_ind = np.argsort(-count)
            sorted_index = ind[count_sort_ind]

            color1 = centroids[sorted_index[0]]
            prop1 = proportions[count_sort_ind[0]]

            if len(ind) > 1:
                color2 = centroids[sorted_index[1]]
                prop2 = proportions[count_sort_ind[1]]
            else:
                color2 = ''
                prop2 = ''   
            #save data
            df_colors_movement.loc[len(df_colors_movement)] = [reg, mov, year, color1, color2, prop1, prop2]

#save csv
df_colors_movement.to_csv("model-output/WikiArtFull_Region_Mov_Year_LAB2D.csv", index=False)

