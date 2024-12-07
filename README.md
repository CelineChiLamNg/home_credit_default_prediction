# Home Credit Risk Prediction
*Celine Ng - December 2024*


This project contains 8 tables originally, a large amount of information. 
My method was to do tables aggregation right after a very brief 
understanding of 
what data/datatype/information was provided, and to only dive deeper into 
what meaning the data was providing afterward. 
However, another approach 
would be to first have a better understanding of the data/tables themselves 
and only select useful/meaning features/columns for aggregation and future 
processes.  <br>
<br>
## Instructions
1. Run the notebooks to create all the necessary files, including the final 
   model.
2. Run the Flask application with:
   python app.py
3. Once server is running, app will be accessible at: http://127.0.0.1:5000

4. The model will return a csv file with the following content: <br>
SK_ID_CURR,predictions <br>
100001,0<br>
100005,0<br>
100013,0<br>
100028,0<br>
100038,1<br>
100042,0<br>
100057,0<br>
...