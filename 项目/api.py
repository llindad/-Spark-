from fastapi import FastAPI
import pandas as pd
#创建FastApi 实例
app=FastAPI()
df=pd.read_csv("recommendations.csv")
#定义一个接口当用户访问时执行
@app.get("/recommend/{user_id}")
def recommend(user_id:int):
	#从表格中筛选出该用户的推荐，取前十条
	user_recs=df[df["user_id"]==user_id]
	#如果没有找到该用户返回空列表
	if user_recs.empty:
	  return{"user_id":user_id,"recommendations":[]}
	#把结果转换成字典列表方便json返回
	recs=user_recs[["item_id","pred_score"]].to_dict(orient="records")
	#返回json 格式的数据
	return{"user_id":user_id,"recommendations":recs}
#直接运行
if __name__=="__main__":
	import uvicorn
	uvicorn.run(app,host="127.0.0.1",port=8000)