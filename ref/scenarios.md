SCENARIO 1: Reason only
  EXAMPLE: ".rewsna eht sa 'tfel' drow eht fo etisoppo eht etirw"
  FLOW:
    no file → agent
    agent: reverse text → read instruction → answer "right"
    no tool call → format_answer
  ANSWER: right

SCENARIO 2: Reason + Python
  EXAMPLE: "Table defines * on S. Which elements break commutativity?"
  FLOW:
    no file → agent
    agent → run_python(check a*b != b*a for all pairs)
    tools → agent: read output → sort, join with ", "
    → format_answer
  ANSWER: b, e

SCENARIO 3: Wikipedia lookup
  EXAMPLE: "Mercedes Sosa studio albums 2000–2009?"
  FLOW:
    no file → agent
    agent → wikipedia("Mercedes Sosa discography")
    tools → agent: count studio albums in range
    → format_answer
  ANSWER: 3

SCENARIO 4: Multi-hop web search
  EXAMPLE: "Actor who played Ray in Polish Raymond → role in Magda M.?"
  FLOW:
    no file → agent
    agent → web_search("Polish Everybody Loves Raymond Ray actor")
    tools → agent: get actor name
    agent → web_search("<actor> Magda M. role")
    tools → agent: get first name only
    → format_answer
  ANSWER: Wojciech

SCENARIO 5: Search + fetch page
  EXAMPLE: "Universe Today article June 6 2023 → paper → NASA award number"
  FLOW:
    agent → web_search(article)
    agent → fetch_page(article_url) → find paper link
    agent → fetch_page(paper_url) → find award number
    → format_answer

SCENARIO 6: YouTube with transcript
  EXAMPLE: "What does Teal'c say to 'Isn't that hot?'"
  FLOW:
    agent → youtube_transcript(url)
    tools → agent: find the line after the question
    → format_answer
  ANSWER: Extremely

SCENARIO 7: YouTube without transcript (visual)
  EXAMPLE: "Highest number of bird species on camera at the same time?"
  FLOW:
    agent → youtube_transcript(url) → no useful data
    agent → web_search(video title + "bird species")
    agent: give best guess
  ANSWER: best guess (accept loss)

SCENARIO 8: Audio file
  EXAMPLE: strawberry pie .mp3 "list the filling ingredients"
  FLOW:
    file → preprocess: transcribe(mp3) → context
    agent: extract ingredients, alphabetize, comma list
    → format_answer
  ANSWER: comma list, alphabetical

SCENARIO 9: Python file
  EXAMPLE: "Final numeric output of attached code?"
  FLOW:
    file → preprocess: read source → context
    agent → run_python(source)       # do not guess, run it
    tools → agent: read last number
    → format_answer
  ANSWER: 0

SCENARIO 10: Excel file
  EXAMPLE: "Total food sales, not drinks, in USD with 2 decimals"
  FLOW:
    file → preprocess: read_excel → markdown table → context
    agent → run_python(pandas: sum food columns)
    tools → agent: format "{:.2f}"
    → format_answer
  ANSWER: 89706.00

SCENARIO 11: Image file
  EXAMPLE: chess position .png, "black to move, winning move"
  FLOW:
    file → preprocess: image → vision message
    agent (vision): read board → FEN
    agent → run_python(stockfish best move)   # optional
    → format_answer
  ANSWER: Rd5   (algebraic notation)

SCENARIO 12: Error handling
  EXAMPLE: tool times out / page 404 / rate limit
  FLOW:
    tools: CATCH error → return "ERROR: ..." to agent
    agent: try other query or other tool
    IF recursion_limit reached → best answer so far
  ANSWER: never empty string
