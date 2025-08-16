from flask import Flask, render_template, request
import task2_script

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            user_input = {
                "date of reservation": request.form.get("reservation_date"),
                "arrival_date": request.form.get("arrival_date"),
                "number of week nights": int(request.form.get("number_of_week_nights") or 0),
                "number of weekend nights": int(request.form.get("number_of_weekend_nights") or 0),
                "special requests": int(request.form.get("special_requests") or 0),
                "no_of_adults": int(request.form.get("adults") or 0),
                "no_of_children": int(request.form.get("children") or 0),
                # إن كان عندك أعمدة إضافية ضمن التدريب أرسلها هنا أيضًا
            }
            out = task2_script.predict(user_input)
            label = out["label"]
            prob  = out["prob"]

            # سمِّ الفئات كما تريد (مثال افتراضي)
            mapping = {0: "Not Booked / Canceled", 1: "Booked / Confirmed"}
            text = mapping.get(label, str(label))
            if prob is not None:
                result = f"Booking Status: {text}  —  Confidence: {prob:.2%}"
            else:
                result = f"Booking Status: {text}"
        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
