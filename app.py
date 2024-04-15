import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

pipe_lr  = joblib.load(open("models/emotion_classifier.pkl","rb"))

def predict_emotion(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


def main():
    st.title("Emotion Classifier App")
    menu = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home - Emotion in Text")

        with st.form(key = "emotion_form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label = "Submit")

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotion(raw_text)
            probability = get_prediction_proba(raw_text)
            
            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Predictions")
                st.write(prediction)
                st.write("Condidence: {}".format(np.max(probability)))
            with col2:
                st.success("Prediction probability")
                proba_df = pd.DataFrame(probability, columns = pipe_lr.classes_)
                clean_df = proba_df.T.reset_index()
                clean_df.columns = ["emotions", "probability"]
                fig = alt.Chart(clean_df).mark_bar().encode(x = "emotions",y = "probability", color = "emotions")
                st.altair_chart(fig, use_container_width = True)


    elif choice == "Monitor":
        st.subheader("Monitor App")
    else:
        st.subheader("About")






if __name__ == "__main__":
    main()