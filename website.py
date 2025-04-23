from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from watermark import MODEL_KEYS, detect_watermark

app = FastAPI()

# Inline HTML for form and results
html_form = """
<!DOCTYPE html>
<html>
  <head>
    <title>AI Text Checker</title>
  </head>
  <body>
    <h1>AI Text Checker</h1>
    <form action="/check" method="post">
      <textarea name="text" rows="10" cols="80" placeholder="Paste your text here"></textarea><br/>
      <button type="submit">Check Text</button>
    </form>
  </body>
</html>
"""

html_result_template = """
<!DOCTYPE html>
<html>
  <head>
    <title>Detection Results</title>
  </head>
  <body>
    <h1>Detection Results</h1>
    <div>{verdict}</div>
    <table border="1" cellpadding="5" cellspacing="0">
      <tr><th>Model ID</th><th>Z-Score</th><th>Flagged?</th></tr>
      {rows}
    </table>
    <p><a href="/">Check another text</a></p>
  </body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def show_form():
    return HTMLResponse(content=html_form)

@app.post("/check", response_class=HTMLResponse)
def check_text(text: str = Form(...)):
    # Simple whitespace tokenization
    tokens = text.strip().split()
    rows = []
    any_flagged = False
    for model_id in MODEL_KEYS:
        z = detect_watermark(model_id, tokens)
        flagged = z > 4.0  # detection threshold
        any_flagged = any_flagged or flagged
        rows.append(f"<tr><td>{model_id}</td><td>{z:.2f}</td><td>{'Yes' if flagged else 'No'}</td></tr>")
    verdict = ("<h2 style='color:red;'>AI-generated text detected!</h2>") if any_flagged else ("<h2>No watermark detected; likely human-written.</h2>")
    html = html_result_template.format(verdict=verdict, rows=''.join(rows))
    return HTMLResponse(content=html)

# To run:
# uvicorn website_demo:app --reload --port 8001
