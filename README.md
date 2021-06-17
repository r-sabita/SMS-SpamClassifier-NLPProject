SMS Spam Classifier
***********************
1. Dataset Info 

- The SMSSpamCollection is a SMS Spam research collected dataset which contains SMS message tagged with 'ham' or 'spam'.

- SMSSpamCollection dataset consists of total of 4827 ham SMS and 747 spam message.
************************
2. Requirements

- pandas
- re
- nltk
- sklearn

*************************
3. Algorithm

step 1: import the required library

step 2: data cleaning and data preprocessing

    step 2.1: create a loop for i in range(0 to len(messages)

    step 2.2: remove everything except a-z|A-Z in the messages

    step 2.3: lower all the letters 

    step 2.4: split the words

    step 2.5 : remove the stopwords and apply stemming in the sentences

    step 2.6: join the message in one line

    step 2.7: add the mgs to corpus
    
step 3: Create Bag Of Words Model

step 4: Do Train Test Split

step 5: Train the model using Naive Bayes Classifier

step 6: Compare the test and predicted data using confusion matrix

step 7: Find accuracy using accuracy_score
******************************

4. Platform

  - IDE (spyder)
  - Language (python)