Data Folder

Directory: /data

Contents:
This directory contains five PDF files with fictional social influencer discovery interviews. Each file provides an in-depth profile of a different social influencer across various niches.

    Milly Adams (@adams_fitness_fanatic) - Focuses on fitness routines and wellness tips.
    Milly Adams (@adams_tech_talk) - Covers the latest technology and gadget reviews.
    Ayesha Patel (@mommybitesandmore) - Offers family-friendly recipes and parenting advice.
    Marco Reyes (@wanderlustwarrior) - Shares travel adventures and destination reviews.
    Naomi Wang (@glowbynaomi) - Provides skincare product reviews and dermatological insights.

These profiles are used within the application to generate tailored advice and insights based on the influencer's focus area.
clean_data.py

Location: /clean_data.py

Functionality:
This Python script is responsible for cleaning and preparing the data from the PDF files for processing. It includes functions to extract text from the PDFs, remove any unwanted characters or formatting, and prepare the data for further analysis or input into the LangChain model.
app.py

Location: /app.py

Functionality:
This is the main Python script that runs the Streamlit application. It integrates the LangChain AI to provide dynamic, conversational AI responses based on the processed influencer profiles. The script includes:

    Setup and initialization of the Streamlit interface.
    Loading and processing of influencer profile data using clean_data.py.
    Implementation of the LangChain conversational model to generate responses.
    Handling user interactions and displaying results in a user-friendly format.

Instructions to Run:
To run the application, navigate to the project's root directory in the terminal and execute the following command:

bash

python3 -m streamlit run app.py

This command launches the Streamlit web server and serves the influencer assistant application, which can be accessed via a web browser at http://localhost:8501.
