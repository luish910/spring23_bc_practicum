import pandas as pd 
from flask import Flask, jsonify, request, abort, Response
import os

#files
directory = os.getcwd()
model_output_path = os.path.join(directory, "model-output")
paintings_csv_name = 'WikiArtFull_LAB2Dcolors.csv'
movements_csv_name = 'WikiArtFull_Region_Mov_Year_LAB2D.csv'
paintings_path = os.path.join(model_output_path, paintings_csv_name)
movements_path = os.path.join(model_output_path, movements_csv_name)

app = Flask(__name__)


df_paintings = pd.read_csv(paintings_path)
df_paintings.columns = ['id', 'year', 'year_group', 'artist', 'title', 'style', 'movement',
       'region', 'rating', 'sentiment', 'colorfulness','url', 'color1', 'color2','prop1', 'prop2']
df_paintings['color1'] = df_paintings['color1'].fillna('')
df_paintings['color2'] = df_paintings['color2'].fillna('')
df_paintings['prop1'] = df_paintings['prop1'].fillna(0.01)
df_paintings['prop2'] = df_paintings['prop2'].fillna(0)
df_paintings['colorfulness'] = df_paintings['colorfulness'].round(decimals = 2)
df_paintings['sentiment'] = df_paintings['sentiment'].round(decimals = 2)

df_movements = pd.read_csv(movements_path)
df_movements.columns =['region', 'movement', 'year', 'color1', 'color2', 'prop1', 'prop2']
df_movements['year_to'] = df_movements['year'].apply(lambda x : x+5)
df_movements['y'] = df_movements[['year', 'year_to']].values.tolist()
df_movements = df_movements.drop(['year_to'], axis=1)

df_movements['color1'] = df_movements['color1'].fillna('')
df_movements['color2'] = df_movements['color2'].fillna('')
df_movements['prop1'] = df_movements['prop1'].fillna(0.01)
df_movements['prop2'] = df_movements['prop2'].fillna(0)
#['Americas', 'Europe', 'MiddleEast', 'AsiaOceania', 'Africa']

@app.route("/movement")
def movement():
    region = request.args.get('region')
    year_from = request.args.get('year_from',type=int)
    year_to = request.args.get('year_to',type=int)
    if year_from == None:
        year_from = 0
    if year_to == None:
        year_to = 3999

    cond_1 = df_movements['region'] == region
    cond_2 = df_movements['year'] >= year_from
    cond_3 = df_movements['year'] <= year_to

    df_response = df_movements[(cond_1)&(cond_2)&(cond_3)]
    response_len = len(df_response)

    try:
        if response_len == 0:
            return '', 204
        else:
            response = jsonify(df_response.to_dict(orient = "records"))
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
    except:
        abort(404)

@app.route("/region")
def region():
    year_from = request.args.get('year_from',type=int)
    year_to = request.args.get('year_to',type=int)
    if year_from == None:
        year_from = 0
    if year_to == None:
        year_to = 3999
    cond_1 = df_paintings['year'] >= year_from
    cond_2 = df_paintings['year'] <= year_to

    df_response = df_paintings[(cond_1)&(cond_2)]
    df_response = pd.DataFrame(df_response['region'].value_counts().reset_index())
    df_response.columns = ['region','count']
    region_palette= ['#a4d3e6','#f5e5c0','#9bc995','#f2b482','#b9887d']
    response_len = len(df_response)
    df_response['fillColor'] = region_palette[0:response_len]
    try:
        if response_len == 0:
            return '', 204
        else:
            response = jsonify(df_response.to_dict(orient = "records"))
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
    except:
        abort(404)

   
@app.route("/paintings")
def paintings():
    region = request.args.get('region')
    year_from = request.args.get('year_from', type = int)
    year_to = request.args.get('year_to', type = int)
    movement = request.args.get('movement')
    img_folder = request.args.get('img_folder')

    #make parameters optional
    if year_from == None:
        year_from = 0
    if year_to == None:
        year_to = 3999  
    if region == None:
        cond_1 = df_paintings['region'] == df_paintings['region'] 
    else:
        cond_1 = df_paintings['region'] == region
    if movement == None:
        cond_4 = df_paintings['movement'] == df_paintings['movement'] 
    else:
        cond_4 = df_paintings['movement'] == movement
    if img_folder == None:
        img_folder = 'images'   
    
    cond_2 = df_paintings['year'] >= year_from
    cond_3 = df_paintings['year'] <= year_to

    df_response = df_paintings[(cond_1)&(cond_2)&(cond_3)&(cond_4)]
    df_response['img_path'] = img_folder+'/'+df_response['id']+'.jpg'
    response_len = len(df_response)
    try:
        if response_len == 0:
            return '', 204
        else:
            response = jsonify(df_response.to_dict(orient = "records"))
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
    except:
        abort(404)

@app.route("/raw_values")
def raw_values():
    response = jsonify(df_paintings.to_dict(orient = "records"))
    return response

@app.route("/hello")
def hello():
    return "Hello, World!"
