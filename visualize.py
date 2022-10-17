import numpy as np
import scipy
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tqdm

import model_train as model_utils

from matplotlib.widgets import Slider, Button


from matplotlib.animation import FuncAnimation
# from IPython import display
import glob
import os

import torch
from PIL import Image

from transformers import AutoFeatureExtractor, AutoModelForImageClassification


from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


import bokeh.io
from bokeh.layouts import layout, gridplot
from bokeh.models import HoverTool, CDSView, ColumnDataSource, IndexFilter, Select, Slider, CustomJS, Div
from bokeh.plotting import figure
from bokeh.plotting import output_notebook, show, output_file, save


def main():
    vis_output_path = "./viz/"

    meta, df, balanced_batsies, preprocess_stage, train_val_0, test_df, kfold_generator = model_utils.load_data()
    val_transforms = preprocess_stage.val_transforms
    preprocess_val_one = preprocess_stage.preprocess_val_one

    model_path = "test_trainer_2022-09-03_23-07-23_9\checkpoint-305"
    fold_idx = int(model_path.split("\\")[0].split("_")[-1])

    fold_idx, (train_df, val_df) = [x for x in kfold_generator if x[0] == fold_idx][0]

    train_ds, val_ds = model_utils.create_dataset(preprocess_stage, train_df, val_df)
    # model = AutoModelForImageClassification.from_pretrained("test_trainer_2022-08-20_19-48-06_3\checkpoint-462").to("cuda:0")
    # model = AutoModelForImageClassification.from_pretrained("test_trainer_2022-09-03_18-47-35_7\checkpoint-60").to("cuda:0")
    model = AutoModelForImageClassification.from_pretrained(model_path).to("cuda:0")
    
    
    
    def predict_one(image, model, val_transforms):
        encoding = torch.stack((val_transforms(image), )).to("cuda:0")
        with torch.no_grad():
            outputs = model(encoding)
            logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        return predicted_class_idx

    def get_df_pred(df2):
        df2['pred'] = df2['image'].progress_apply(predict_one, model=model, val_transforms=preprocess_val_one)
        df2['correct'] = df2['pred'] == df2['rural'].astype(int)
        return df2
    test_df = get_df_pred(test_df)
    val_df = get_df_pred(val_df)
    train_df = get_df_pred(train_df)
    
    

    # model.config_class.update({"use_return_dict": False})


    # target_layers = [model.layer4[-1]]
    target_layers = [model.convnext.encoder.stages[-1].layers[-1].dwconv]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    # targets = [ClassifierOutputTarget(0)]

    def wrapper(func):
        def model_forward_decorator(*args, **kargs):
            res = func(*args, **kargs)
            return getattr(res, "logits", res)
        return model_forward_decorator

    model.forward = wrapper(model.forward)


    # idx = 161
    # image = test_df.iloc[idx]['image']
    def get_grayscale(row):
        image = row['image']
        rgb_img = image.point(lambda p: p * ((2 ** 8) / (2 ** 16)), mode='RGB').convert("RGB")
        rgb_img = rgb_img.resize((224, 224))
        rgb_img = np.float32(rgb_img) / 255
        label = int(row['rural'])
    # print("label", label)
        encoding = torch.stack((preprocess_val_one(image),)).to("cuda:0")

        tensor = encoding  # .unsqueeze(0)


        input_tensor = tensor.cuda()
        targets = [ClassifierOutputTarget(label)]
        # targets = [HuggingFaceClassifierOutputTarget(1)]
        # targets = None
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        return rgb_img, grayscale_cam


    def get_visualization(row):
        rgb_img, grayscale_cam = get_grayscale(row)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        return pd.Series([visualization, rgb_img, grayscale_cam], index=['grad_cam_visualizations', 'grad_cam_rgb_img', 'grad_cam_grayscale_cam'])

    test_df[['grad_cam_visualizations', 'grad_cam_rgb_img', 'grad_cam_grayscale_cam']] = test_df.apply(get_visualization, axis=1)
    
    
    def save_visualization(row, df_name):
        dir_path = vis_output_path + f"./images/{df_name}/rural_{row['rural']}/bat_id_{row['bat_id']}/"
        name = f"idx_{row['image_index']}.png"
        im = Image.fromarray(row['grad_cam_visualizations'])
        os.makedirs(dir_path, exist_ok=True)
        im.save(dir_path + name)
    test_df.progress_apply(save_visualization, df_name="test_df", axis=1)
    
    
    
    
        
    # Plot
    select_bat_id = test_df.query("rural == True")['bat_id'].unique()[-1]
    selected_bat_df = test_df.query("bat_id == @select_bat_id")
    select_rural = selected_bat_df['rural'].iloc[0]
    show_df2 = selected_bat_df['grad_cam_visualizations']

    # plot = figure(title=f"Bat ID {select_bat_id}, rural={select_rural}")


    # def get_bokeh_img(src):

    #     rgb = src
    #     rgba = np.dstack((rgb, 255*np.ones(rgb.shape[:-1])))

    #     M, N = rgba.shape[:2]
    #     img = np.empty((M, N), dtype=np.uint32)
    #     view = img.view(dtype=np.uint8).reshape((M, N, 4))
    #     view[:,:,0] = rgba[:,:,0] # copy red channel
    #     view[:,:,1] = rgba[:,:,1] # copy blue channel
    #     view[:,:,2] = rgba[:,:,2] # copy green channel
    #     view[:,:,3] = 255

    #     img = img[::-1] # flip for Bokeh

    #     return img

    start_index = 45

    # bokeh_df = pd.DataFrame({"image": show_df2.apply(get_bokeh_img)}).reset_index()
    # source = ColumnDataSource(data={'image': [bokeh_df['image'][start_index]]})
    # view = CDSView(source=source, filters=[IndexFilter([0])])
    # test_df
    def get_img_url(row, df_name):
        dir_path = f"./images/{df_name}/rural_{row['rural']}/bat_id_{row['bat_id']}/"
        name = f"idx_{row['image_index']}.png"
        return dir_path + name

    best_indexes = list(range(100))

    # bokeh_df = pd.DataFrame({"image": test_df.apply(get_img_url, axis=1, df_name="test_df")}).reset_index()
    # bokeh_df = test_df[['rural', 'bat_id', 'correct']].assign(image_url=test_df.apply(get_img_url, axis=1, df_name="test_df"))
    bokeh_df = test_df[['rural', 'bat_id', 'correct', 'image_index']].assign(image_url=test_df.apply(get_img_url, axis=1, df_name="test_df"), is_best_index=test_df['image_index'].isin(best_indexes))#.groupby(["rural", )
    bokeh_df_dict = bokeh_df.groupby("rural").apply(lambda x: x.groupby("bat_id").apply(lambda y: y.groupby("image_index")[['correct', 'image_url', 'is_best_index']].first().to_dict(orient='index')).to_dict()).to_dict()
    source = ColumnDataSource(data={'image': [bokeh_df_dict[True][select_bat_id][start_index]['image_url']]})
    source1 = ColumnDataSource(data={'image': [bokeh_df_dict[True][select_bat_id][start_index]['image_url']]})
    source2 = ColumnDataSource(data={'image': [bokeh_df_dict[True][select_bat_id][start_index]['image_url']]})
    source3 = ColumnDataSource(data={'image': [bokeh_df_dict[True][select_bat_id][start_index]['image_url']]})

    rural_to_bats_map = test_df.groupby("rural")['bat_id'].unique().apply(list).to_dict()
    # source = ColumnDataSource(data=(dict(image=[get_bokeh_img(show_df2.iloc[46])],
    #                                     x=[0],
    #                                     y=[0],
    #                                     dw=[10],
    #                                     dh=[10])))

    # bokeh_im = plot.image_rgba(source=source, image='image', x=0, y=0, dw=10, dh=10
    #                 )

    # cb = CustomJS(args=dict(graph=source, source=bokeh_df['image'].to_dict()), code="""
    #                 graph.data['image'][0] = source[cb_obj.value];
    #             graph.change.emit();
    # """)


    plot = figure(title=f"Test bats")
    bokeh_im = plot.image_url(source=source, url='image', x=0, y=0, w=10, h=10
                    )
    correct_text = Div(text="", align="center")

    plot1 = figure(title=f"Test bats")
    bokeh_im1 = plot1.image_url(source=source1, url='image', x=0, y=0, w=10, h=10
                    )
    correct_text1 = Div(text="", align="center")

    plot2 = figure(title=f"Test bats")
    bokeh_im2 = plot2.image_url(source=source2, url='image', x=0, y=0, w=10, h=10
                    )
    correct_text2 = Div(text="", align="center")

    plot3 = figure(title=f"Test bats")
    bokeh_im3 = plot3.image_url(source=source3, url='image', x=0, y=0, w=10, h=10
                    )
    correct_text3 = Div(text="", align="center")

    # Slider widget
    # bat_indexes_str = [str(x['bat_id']) + "-" + str(x['rural']) for _, x in test_df[['bat_id', "rural"]].drop_duplicates().iterrows()]
    # select_bats = Select(title="Bat index:", value=bat_indexes_str[0], options=bat_indexes_str)
    bat_indexes_str = [str(x['rural']) for _, x in test_df[["rural"]].drop_duplicates().iterrows()]
    select_bats = Select(title="Bat is rural:", value=bat_indexes_str[0], options=bat_indexes_str)


    # Slider widget
    slider = Slider(start=bokeh_df['image_index'].min(), end=bokeh_df['image_index'].max(), step=1, value=start_index, title='Z axis')


    cb = CustomJS(args=dict(bokeh_df_dict=bokeh_df_dict, select_bats=select_bats, slider=slider, 
                            rural_to_bats_map=rural_to_bats_map,
                            graph=source, correct_text=correct_text, 
                            graph1=source1, correct_text1=correct_text1, 
                            graph2=source2, correct_text2=correct_text2, 
                            graph3=source3, correct_text3=correct_text3), code="""
                var img_idx = slider.value;
                
                // var bat_id = select_bats.value.split("-");
                // var bat_idx = bat_id[0];
                // var bat_type = bat_id[1];
                // var bat_type_bool = (bat_type === 'True');
                var bat_type_bool = (select_bats.value === 'True');
                
                function set_graph(btype, bidx, g, c) {
                    var dir_path = bokeh_df_dict[btype][bidx][img_idx]['image_url'];
                    g.data['image'][0] = dir_path;
                    c.text = "<b>Bat index:</b> <u>" + bidx + "</u>; <b>correct:</b> " + bokeh_df_dict[btype][bidx][img_idx]['correct'] + "; " //<b>confident index:</b> " + bokeh_df_dict[btype][bidx][img_idx]['is_best_index'];
                    g.change.emit();
                }
                
                set_graph(bat_type_bool, rural_to_bats_map[bat_type_bool][0], graph, correct_text)
                set_graph(bat_type_bool, rural_to_bats_map[bat_type_bool][1], graph1, correct_text1)
                set_graph(bat_type_bool, rural_to_bats_map[bat_type_bool][2], graph2, correct_text2)
                set_graph(bat_type_bool, rural_to_bats_map[bat_type_bool][3], graph3, correct_text3)
    """)

    select_bats.js_on_change("value", cb)
    # (CustomJS(graph=source, code="""
    #             graph.change.emit();
    #     # console.log('select: value=' + this.value, this.toString())
    # """))
    slider.js_on_change('value', cb)

    # show(plot, allow_websocket_origin="*")
    # prepare some data
    # x = [1, 2, 3, 4, 5]
    # y = [4, 5, 5, 7, 2]
    # circle = plot.circle(x, y, fill_color="red", size=15)

    l = layout([[select_bats], 
                [slider],
                gridplot(
                        [[correct_text, correct_text1],
                        [plot, plot1]]),
                gridplot(
                        [[correct_text2, correct_text3],
                        [plot2, plot3]])
                ])


    output_file(filename=vis_output_path + f"test_df_bats.html", title="Static HTML file")
    # curdoc().add_root(l)
    show(l)
    # output_notebook(l)
    

if __name__ == "__main__":
    main()
