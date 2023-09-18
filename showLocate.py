import os,argparse
import random
        
from fire import initFire, FireModel, FireRunner, FireData

from config import cfg
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

def drawLocate(output_path, res_dict):
    img_size = (500, 500) # 图片大小
    blue_color = (0, 0, 255) # 蓝色
    red_color = (255, 0, 0) # 红色
    
    img = Image.new('RGB', img_size, color='white') # 创建空白图像
    draw = ImageDraw.Draw(img) # 创建画图对象
    
    # 画坐标轴
    axis_width = 2 # 坐标轴宽度
    draw.line([(0, img_size[1]//2), (img_size[0], img_size[1]//2)], fill='black', width=axis_width)
    draw.line([(img_size[0]//2, 0), (img_size[0]//2, img_size[1])], fill='black', width=axis_width)
    
    font_size = 16 # 字体大小
    font = ImageFont.truetype("Arial.ttf", font_size) # 加载字体文件
    tick_length = 5 # 刻度长度
    axis_range = 2 # 坐标轴范围
    
    # 添加刻度值
    for i in range(-axis_range, axis_range+1):
        x = int((i+axis_range)*img_size[0]/(axis_range*2))
        draw.line([(x, img_size[1]//2-tick_length), (x, img_size[1]//2+tick_length)], fill='black', width=axis_width)
        draw.text((x-font_size//2, img_size[1]//2+tick_length), str(i), font=font, fill='black')
        
    for i in range(-axis_range, axis_range+1):
        y = int((axis_range-i)*img_size[1]/(axis_range*2))
        draw.line([(img_size[0]//2-tick_length, y), (img_size[0]//2+tick_length, y)], fill='black', width=axis_width)
        draw.text((img_size[0]//2+tick_length, y-font_size//2), str(i), font=font, fill='black')
    
    # 添加坐标点
    for img_name, coords in res_dict.items():
        y, x0, y0 = coords
        color = blue_color if y == 0 else red_color
        x, y = int((x0*img_size[0]/4)+img_size[0]//2), int((y0*img_size[1]/4)++img_size[1]//2)
        draw.point((x, y), color) # 在指定坐标上画点
        
    img.save(output_path) # 保存图像到指定路径




def predict(cfg):

    initFire(cfg)


    model = FireModel(cfg)
    
    

    data = FireData(cfg)
    # data.showTrainData()
    # b
    
    test_loader = data.getEvalDataloader()


    runner = FireRunner(cfg, model)

    #print(model)
    runner.modelLoad(cfg['model_path'])


    res_dict = runner.locate(test_loader)
    #print(res_dict)
    print(len(res_dict)) #{name:label, x0,y0}
    
    drawLocate("output/draw.jpg", res_dict)



def predictMerge(cfg):
    initFire(cfg)


    model = FireModel(cfg)
    
    

    data = FireData(cfg)
    # data.showTrainData()
    # b
    
    test_loader = data.getTestDataloader()
    runner1 = FireRunner(cfg, model)
    runner1.modelLoad('output/efficientnet-b6_e17_fold0_0.93368.pth')
    print("load model1, start running.")
    res_dict1 = runner1.predictRaw(test_loader)
    print(len(res_dict1))

    test_loader = data.getTestDataloader()
    runner2 = FireRunner(cfg, model)
    runner2.modelLoad('output/efficientnet-b6_e18_fold1_0.94537.pth')
    print("load model2, start running.")
    res_dict2 = runner2.predictRaw(test_loader)

    test_loader = data.getTestDataloader()
    runner3 = FireRunner(cfg, model)
    runner3.modelLoad('output/efficientnet-b6_e14_fold2_0.91967.pth')
    print("load model3, start running.")
    res_dict3 = runner3.predictRaw(test_loader)

    test_loader = data.getTestDataloader()
    runner4 = FireRunner(cfg, model)
    runner4.modelLoad('output/efficientnet-b6_e18_fold3_0.92239.pth')
    print("load model4, start running.")
    res_dict4 = runner4.predictRaw(test_loader)

    # test_loader = data.getTestDataloader()
    # runner5 = FireRunner(cfg, model)
    # runner5.modelLoad('output/efficientnet-b6_e17_fold0_0.93368.pth')
    # print("load model5, start running.")
    # res_dict5 = runner5.predictRaw(test_loader)


    res_dict = {}
    for k,v in res_dict1.items():
        #print(k,v)
        v1 =np.argmax(v+res_dict2[k]+res_dict3[k]+res_dict4[k])
        res_dict[k] = v1
    
    res_list = sorted(res_dict.items(), key = lambda kv: int(kv[0].split("_")[-1].split('.')[0]))
    print(len(res_list), res_list[0])

    # to csv
    # res_list_final = []
    # for res in res_list:
    #     res_list_final.append([res[0]]+res[1])
    # #res_df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['type'])
    # #res_df = res_df.reset_index().rename(columns={'index':'id'})
    # res_df = DataFrame(res_list_final, columns=['id','type','color','toward'])
    

    # res_df.to_csv(os.path.join(cfg['save_dir'], 'result.csv'), 
    #                             index=False,header=True)

    with open('result.csv', 'w', encoding='utf-8') as f:
        f.write('file,label\n')
        for i in range(len(res_list)):
            line = [res_list[i][0], str(res_list[i][1])]
            line = ','.join(line)
            f.write(line+"\n")


def predictTTA(cfg):

    pass


def predictMergeTTA(cfg):

    pass


def main(cfg):

    if cfg["merge"]:
        if cfg["TTA"]:
            predictMergeTTA(cfg)
        else:
            predictMerge(cfg)
    else:
        if cfg["TTA"]:
            predictTTA(cfg)
        else:
            predict(cfg)


    



if __name__ == '__main__':
    main(cfg)