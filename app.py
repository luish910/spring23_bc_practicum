import pandas as pd 
from flask import Flask, jsonify, request, abort, Response
import os

#files
directory = os.getcwd()
model_output_path = os.path.join(directory, "model-output")
paintings_csv_name = 'WikiArtFull_colors.csv'
movements_csv_name = 'WikiArtFull_Region_Mov_Year.csv'
paintings_path = os.path.join(model_output_path, paintings_csv_name)
movements_path = os.path.join(model_output_path, movements_csv_name)

app = Flask(__name__)

df_paintings = pd.read_csv(paintings_path)
df_paintings.columns = ['id', 'year', 'year_group', 'artist', 'title', 'style', 'movement',
       'region', 'rating', 'sentiment', 'colorfulness',
       'url', 'color1', 'color2', 'prop1', 'prop2']

df_movements = pd.read_csv(movements_path)
df_movements.columns =['region', 'movement', 'year', 'color1', 'color2', 'prop1', 'prop2']
df_movements['region'].unique()
#['Americas', 'Europe', 'MiddleEast', 'AsiaOceania', 'Africa']

@app.route("/movement")
def movement():
    region = request.args.get('region')
    year_from = request.args.get('year_from')
    year_to = request.args.get('year_to')

    year_from = int(year_from)
    year_to = int(year_to)

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
    year_from = request.args.get('year_from')
    year_to = request.args.get('year_to')
    year_from = int(year_from)
    year_to = int(year_to)

    cond_1 = df_paintings['year'] >= year_from
    cond_2 = df_paintings['year'] <= year_to

    df_response = df_paintings[(cond_1)&(cond_2)]
    df_response = pd.DataFrame(df_response['region'].value_counts().reset_index())
    df_response.columns = ['region','count']
    response_len = len(df_response)
    region_palette= ['#a4d3e6','#f5e5c0','#9bc995','#f2b482','#b9887d']
    df_response['fillColor'] = region_palette[0:region_palette]
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
    year_from = request.args.get('year_from')
    year_to = request.args.get('year_to')
    movement = request.args.get('movement')
    img_folder = request.args.get('img_folder')

    year_from = int(year_from)
    year_to = int(year_to)
    
    cond_1 = df_paintings['region'] == region
    cond_2 = df_paintings['year'] >= year_from
    cond_3 = df_paintings['year'] <= year_to
    cond_4 = df_paintings['movement'] == movement

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


@app.route("/")
def hello():
    return "Hello, World!"
