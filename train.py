import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 🧠 Training data
data = [
    # Food
    ("zomato", "Food"), ("swiggy", "Food"), ("mcdonalds", "Food"),
    ("kfc", "Food"), ("pizza hut", "Food"), ("dominos", "Food"),
    ("burger king", "Food"), ("restaurant", "Food"), ("cafe", "Food"),
    ("coffee", "Food"), ("lunch", "Food"), ("dinner", "Food"),
    ("breakfast", "Food"), ("grocery", "Food"), ("vegetables", "Food"),
    ("fruits", "Food"), ("milk", "Food"), ("bread", "Food"),
    ("eggs", "Food"), ("rice", "Food"), ("dal", "Food"),
    ("biryani", "Food"), ("chai", "Food"), ("snacks", "Food"),
    ("blinkit", "Food"), ("bigbasket", "Food"), ("dunzo", "Food"),
    ("instamart", "Food"), ("zepto", "Food"), ("dmart", "Food"),

    # Transport
    ("uber", "Transport"), ("ola", "Transport"), ("rapido", "Transport"),
    ("auto", "Transport"), ("bus", "Transport"), ("metro", "Transport"),
    ("train", "Transport"), ("flight", "Transport"), ("petrol", "Transport"),
    ("diesel", "Transport"), ("fuel", "Transport"), ("cab", "Transport"),
    ("taxi", "Transport"), ("rickshaw", "Transport"), ("bike", "Transport"),
    ("irctc", "Transport"), ("indigo", "Transport"), ("air india", "Transport"),
    ("redbus", "Transport"), ("parking", "Transport"), ("toll", "Transport"),

    # Shopping
    ("amazon", "Shopping"), ("flipkart", "Shopping"), ("myntra", "Shopping"),
    ("ajio", "Shopping"), ("meesho", "Shopping"), ("nykaa", "Shopping"),
    ("clothes", "Shopping"), ("shoes", "Shopping"), ("shirt", "Shopping"),
    ("jeans", "Shopping"), ("dress", "Shopping"), ("accessories", "Shopping"),
    ("watch", "Shopping"), ("bag", "Shopping"), ("wallet", "Shopping"),
    ("electronics", "Shopping"), ("mobile", "Shopping"), ("laptop", "Shopping"),
    ("headphones", "Shopping"), ("charger", "Shopping"), ("books", "Shopping"),

    # Health
    ("medicine", "Health"), ("pharmacy", "Health"), ("doctor", "Health"),
    ("hospital", "Health"), ("clinic", "Health"), ("gym", "Health"),
    ("fitness", "Health"), ("yoga", "Health"), ("apollo", "Health"),
    ("medplus", "Health"), ("1mg", "Health"), ("netmeds", "Health"),
    ("protein", "Health"), ("vitamins", "Health"), ("supplements", "Health"),
    ("dental", "Health"), ("eye care", "Health"), ("blood test", "Health"),
    ("diagnostic", "Health"), ("physiotherapy", "Health"),

    # Entertainment
    ("netflix", "Entertainment"), ("hotstar", "Entertainment"),
    ("amazon prime", "Entertainment"), ("spotify", "Entertainment"),
    ("youtube", "Entertainment"), ("movie", "Entertainment"),
    ("cinema", "Entertainment"), ("pvr", "Entertainment"),
    ("inox", "Entertainment"), ("concert", "Entertainment"),
    ("gaming", "Entertainment"), ("steam", "Entertainment"),
    ("playstation", "Entertainment"), ("xbox", "Entertainment"),
    ("book", "Entertainment"), ("kindle", "Entertainment"),
    ("apple music", "Entertainment"), ("zee5", "Entertainment"),
    ("sonyliv", "Entertainment"), ("jiocinema", "Entertainment"),

    # Bills
    ("electricity", "Bills"), ("water bill", "Bills"), ("gas bill", "Bills"),
    ("internet", "Bills"), ("wifi", "Bills"), ("broadband", "Bills"),
    ("mobile recharge", "Bills"), ("postpaid", "Bills"), ("prepaid", "Bills"),
    ("rent", "Bills"), ("maintenance", "Bills"), ("society", "Bills"),
    ("insurance", "Bills"), ("lic", "Bills"), ("emi", "Bills"),
    ("loan", "Bills"), ("credit card", "Bills"), ("tax", "Bills"),
    ("jio", "Bills"), ("airtel", "Bills"), ("bsnl", "Bills"),

    # Other
    ("gift", "Other"), ("donation", "Other"), ("charity", "Other"),
    ("stationery", "Other"), ("pen", "Other"), ("notebook", "Other"),
    ("salon", "Other"), ("haircut", "Other"), ("spa", "Other"),
    ("laundry", "Other"), ("repair", "Other"), ("plumber", "Other"),
    ("electrician", "Other"), ("cleaning", "Other"), ("miscellaneous", "Other"),
    ("atm", "Other"), ("cash", "Other"), ("transfer", "Other"),
]

# Split data
titles = [d[0] for d in data]
categories = [d[1] for d in data]

X_train, X_test, y_train, y_test = train_test_split(
    titles, categories, test_size=0.15, random_state=42
)

# 🤖 Build and train the model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        analyzer="char_wb",
        min_df=1,
        sublinear_tf=True
    )),
    ("clf", MultinomialNB(alpha=0.1)),
])

model.fit(X_train, y_train)

# 📊 Print accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained! Accuracy: {accuracy * 100:.1f}%")

# Test predictions
test_cases = ["uber", "netflix", "zomato", "electricity", "amazon", "gym", "haircut"]
print("\n🧪 Test predictions:")
for t in test_cases:
    print(f"  '{t}' → {model.predict([t])[0]}")

# 💾 Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n💾 Model saved as model.pkl")