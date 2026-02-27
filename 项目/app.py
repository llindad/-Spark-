import streamlit as st
import requests
import pandas as pd
import altair as alt
#设置网页标题和布局
st.set_page_config(page_title="电商推荐系统",layout="wide")
st.title("基于Spark的电商用户消费偏好计算系统")
#在左边栏添加一个输入框（让用户输入id)
with st.sidebar:
	st.header("用户输入")
	user_id=st.number_input("请输入用户id",min_value=1,value=123,step=1)
	api_url=st.text_input("API地址",value="http://127.0.0.1:8000")
#主区域：当用户点按钮时，调用API显示结果
if  st.button("获取推荐"):
	with st.spinner("正在获取推荐..."):
	   try:
	   #调用后端ＡＰＩ
	      response=requests.get(f"{api_url}/recommend/{user_id}")
	      if response.status_code==200:
	          data =response.json()
	          recs=data["recommendations"]
	          if recs:
	             #把json转换成pandas DataFrame，方便显示
	             df=pd.DataFrame(recs)
	             st.subheader(f"用户{user_id}的推荐用品")#显示表格
	             st.subheader("推荐评分分布")
	             chart=alt.Chart(df).mark_bar().encode(
		x=alt.X('item_id:N',title='商品 id'),
		y=alt.Y('pred_score:Q',title='预测评分'),
		color=alt.value('#FFD700')
	              ).properties(
		     width=600,height=400
	                )
	             st.altair_chart(chart,use_container_width=True)            
	          else:
	             st.warning(f"用户{user_id}暂无推荐")
	      else:
	           st.error(f"API返回错误：{response.status_code}")
	   except Exception as e:
	          st.error(f"连接失败：{e}")
#页脚说明
st.markdown("---")
st.caption("说明：后端API 需先启动，请确保api.py正在运行。")

