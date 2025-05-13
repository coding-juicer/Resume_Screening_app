import streamlit as st
import pickle
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')

clf = pickle.load(open('.venv/clf.pkl', 'rb'))
tfidf = pickle.load(open('.venv/tfidf.pkl', 'rb'))#vectorization


def clearnResume(txt):
    cleanTxt = re.sub(r'http\S+'," ",txt)
    cleanTxt = re.sub(r'RT|CC'," ",cleanTxt)
    cleanTxt = re.sub(r'#\S+\s'," ",cleanTxt)
    cleanTxt = re.sub(r'@\S+'," ",cleanTxt)
    cleanTxt = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[/]^_`{|}~""")," ",cleanTxt)#speasal character
    cleanTxt = re.sub(r'[^\x00-\x7f]'," ",cleanTxt)
    cleanTxt = re.sub(r'\s'," ",cleanTxt)
    return cleanTxt

def main():
    st.title('Resume Screening app')

    uploaded_file = st.file_uploader("Choose a file",type=['pdf','txt'])
    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')#it convert binary to txt
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        clearn_resume = clearnResume(resume_text)

        clearn_resume = tfidf.transform([clearn_resume])
        predicted_id = clf.predict(clearn_resume)[0]
        st.write(predicted_id)
        category_mapping = {
            6: 'Data Science',
            12: "HR",
            0: "Advocate",
            1: "Arts",
            24: "Web Designing",
            16: "Mechanical Engineer",
            22: "Sales",
            14: "Health and fitness",
            5: "Civil Engineer",
            15: "Java Developer",
            4: "Business Analyst",
            21: "SAP Developer",
            2: "Automation Testing",
            11: "Electrical Engineering",
            18: "Operations Manager",
            20: "Python Developer",
            8: "DevOps Engineer",
            17: "Network Security Engineer",
            19: "PMO",
            7: "Database",
            13: "Hadoop",
            10: "ETL Developer",
            9: "DotNet Developer",
            3: "Blockchain",
            23: "Testing"
        }
        category_name = category_mapping.get(predicted_id, "Unknown")
        st.write(category_name)






# python main
if __name__ == '__main__':
    main()