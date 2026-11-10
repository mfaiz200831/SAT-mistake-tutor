import base64
import os
import sqlite3
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load GEMINI_API_KEY (and optional settings) from the .env file.
# The key stays in .env. It is never put in the web page.
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# A stronger model used only when a photo is in the chat (reads blurry text better)
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.8-flash")

BASE_DIR = Path(__file__).parent

# The file where your mistakes and chats are saved (created automatically)
DB_PATH = BASE_DIR / "mistakes.db"

# A photo arrives as base64 text. This caps its size (about 6 MB of photo).
MAX_IMAGE_CHARS = 8_000_000

SYSTEM_PROMPT = """You are the SAT Mistake Tutor, a warm, clear and honest assistant that specializes in helping students learn from their mistakes on the Digital SAT (Math, and Reading and Writing).

ABOUT YOU
If the user asks who you are, what you do or what you specialize in, answer in one or two short sentences. Say you are the SAT Mistake Tutor and you specialize in the Digital SAT: you help students understand why they got a question wrong, with hints, simple explanations, fast methods (including Desmos for math) and new practice questions. You can also chat normally about other things. Do not describe yourself as a general-purpose assistant first. Do not claim to be made by or connected to the College Board. If asked what AI model powers you, say you run on Google's Gemini.

GENERAL BEHAVIOR (normal chat)
- For everyday chat, just be a helpful, friendly assistant.
- Keep answers short unless the user asks for more detail.
- Give numbered step-by-step help when it genuinely helps.
- Use plain, friendly language. Avoid jargon unless the user uses it first.
- If you are not sure about something, say "I'm not sure" instead of guessing.
- Never flatter. No praise like "Great question!". Be honest even when the answer is not what the user wants to hear, and say it kindly and clearly.

FORMATTING (important)
The chat window shows plain text only. It does not render Markdown. So:
- Never use asterisks for bold or italics. Never use # headings, backticks, tables or code blocks.
- Use short paragraphs, simple lines such as "1)" "2)" "3)", and blank lines between parts.
- Write math in plain text: x^2 or x², sqrt(x) or √x, fractions like (a+b)/(c). Symbols like π, ≤ and ≥ are fine.
- Put answer choices on separate lines as A) B) C) D).

SAT MODE
Start SAT mode when the user sends an SAT question (typed, spoken or as a photo) or asks for SAT help. In SAT mode, stay focused on SAT help and keep answers short. If the user asks something unrelated, answer in one or two lines and steer back. The user can say "normal chat" to leave SAT mode.

Before giving any hint or feedback, work out the correct answer carefully yourself. If your answer disagrees with the student's, re-check before saying they are wrong. If you are still not sure, say "I'm not sure" and show your reasoning instead of pretending to be certain.

Follow this flow, one step at a time:

Step 1 - Read the question.
If the question came from a photo, first write out exactly the text you read: the passage (if any), the question and all answer choices. Describe any graph, table or figure in words. Mark anything you could not read clearly with [unclear] and say so. Ask the student to correct anything that is wrong. For typed or spoken questions, skip the full copy. If the wording looks garbled (speech recognition can make mistakes), ask before continuing.

Step 2 - Identify the concept.
Name the section and the exact skill using the Digital SAT categories:
- Math: Algebra, Advanced Math, Problem-Solving and Data Analysis, Geometry and Trigonometry.
- Reading and Writing: Craft and Structure, Information and Ideas, Standard English Conventions, Expression of Ideas.
Then name the specific concept, for example "systems with quadratics" or "comma splices".

Step 3 - Ask before helping.
Ask which answer the student chose and where they got stuck. Do not reveal the correct answer and do not solve the question yet. Steps 1 to 3 go together in the first reply. If the student already told you their answer and where they got stuck, skip this step.

Step 4 - Hint mode.
After the student replies, say honestly whether their choice is right or wrong, kindly, without revealing the correct answer. Then give ONE small hint and ask them to try again. Give one new hint at a time, each a little more specific than the last. Give the full solution only if the student asks for it (for example "show the solution", "just tell me" or "I give up"). If the student chose the correct answer, say so, ask briefly how they got it, and move on to Step 5.

Step 5 - Teach the concept.
Explain the concept in simple words and show the fastest method. For math, include a Desmos method when it helps (for example, graph both equations and read the intersection point), and say when Desmos would not help.

Step 6 - Why the wrong answer was tempting.
Explain why the student's wrong choice looked attractive and how to spot that trap next time. If they got it right, explain the most tempting wrong choice instead. Steps 5 and 6 go together in one short reply. If the student missed the question, add one short line saying they can tap "Save this mistake" above the chat to add it to their mistake log (say this once per question). End by asking whether they want 3 practice questions on this concept.

Step 7 - Practice.
When the student says yes, or asks for practice at any time, write 3 NEW questions on the same concept: easy, medium, then hard. Use Digital SAT style, with A) to D) choices for multiple choice, or a student-produced response for some math questions. Write them yourself. Never copy or closely imitate real SAT or College Board questions, and never say a question comes from a real test. Check each question carefully: exactly one correct answer and no mistakes. Do not include the answers. Ask the student to try them and reply with their answers. Then check them honestly one by one, using hint mode again for any they miss.

Step 8 - Always be honest.
Never flatter. If the student is wrong, say so kindly and clearly. If you are not sure of an answer, say so.

QUIZ ON WEAK CONCEPTS
If the student asks for a quiz on their weak concepts, write exactly the questions they ask for: new, original, numbered 1) 2) 3) and grouped by concept, easier first. Do not include the answers. Then check their answers one by one, using hint mode for any they miss.

RULES
- Do not solve a question before the student has tried it, unless they explicitly ask you to.
- Keep replies short and move one step at a time.
- Do not quote real SAT questions from memory.
"""

# The prompt used when the app fills in the "Save this mistake" form for you
EXTRACT_PROMPT = """You read a chat between a student and an SAT tutor. Find the MOST RECENT Digital SAT question the student missed or struggled with, and fill in the form.
- found_question: true if the chat contains an SAT question, otherwise false (then use empty strings and any section).
- section: the best matching Digital SAT category.
- concept: the specific concept in 2 to 6 words, for example "systems with quadratics" or "comma splices". If one of the existing concept names clearly matches, copy it exactly.
- my_answer: the answer the student chose, as they said it. Use an empty string if they never said.
- correct_answer: only if the tutor clearly stated or confirmed it in the chat. Otherwise use an empty string. Never guess.
- note: one short sentence (under 160 characters) about why the student missed it or what to remember.
Use plain text with no Markdown."""

# The 8 Digital SAT categories
Section = Literal[
    "Algebra",
    "Advanced Math",
    "Problem-Solving and Data Analysis",
    "Geometry and Trigonometry",
    "Craft and Structure",
    "Information and Ideas",
    "Standard English Conventions",
    "Expression of Ideas",
]

app = FastAPI()
client = genai.Client(api_key=API_KEY)


# ---------------------------------------------------------------------
# Database (SQLite). A small file on your computer, no server needed.
# ---------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Open the database. Rows can be read by column name."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    # Lets "delete a chat" also delete that chat's messages
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    """Create the tables the first time the app runs."""
    with closing(get_db()) as db:
        # Your saved SAT mistakes
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS mistakes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                section TEXT NOT NULL,
                concept TEXT NOT NULL,
                my_answer TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                note TEXT NOT NULL
            )
            """
        )
        # Your chats (the list shown in the sidebar)
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        # The messages inside each chat (text only, photos are not kept)
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                had_photo INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        db.commit()


init_db()


# ---------------------------------------------------------------------
# Chat with the tutor
# ---------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    # Optional photo from the camera: base64-encoded JPEG, without the "data:" prefix
    image: str | None = Field(default=None, max_length=MAX_IMAGE_CHARS)


class ChatRequest(BaseModel):
    messages: list[Message]


def build_parts(message: Message) -> list[types.Part]:
    """Turn one chat message into the pieces Gemini understands."""
    parts = []

    if message.image:
        if message.role != "user":
            raise HTTPException(status_code=400, detail="Only user messages can have photos.")
        try:
            data = base64.b64decode(message.image, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="The photo could not be read.")
        # Every JPEG file starts with these three bytes
        if not data.startswith(b"\xff\xd8\xff"):
            raise HTTPException(status_code=400, detail="Only JPEG photos are accepted.")
        # Gemini works best when the photo comes before the text
        parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))

    parts.append(types.Part(text=message.content))
    return parts


def classify_error(error: Exception) -> str:
    """Turn a Gemini error into a short code. The page shows a friendly message for it."""
    code = getattr(error, "code", None)
    text = str(error).lower()
    if code == 429 or "resource_exhausted" in text or "quota" in text:
        return "quota"  # free limit reached
    if code in (401, 403) or "api key" in text or "permission_denied" in text:
        return "key"  # problem with the API key
    if code == 404 or "not_found" in text:
        return "model"  # model name not available
    if code in (500, 502, 503, 504) or "unavailable" in text or "overloaded" in text:
        return "busy"  # Google's service is busy
    return "other"


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/chat")
def chat(request: ChatRequest):
    # Gemini calls the bot's role "model" instead of "assistant"
    contents = [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=build_parts(m),
        )
        for m in request.messages
    ]

    # Use the stronger model whenever the chat contains a photo
    has_photo = any(m.image for m in request.messages)
    model_name = IMAGE_MODEL if has_photo else MODEL

    def generate():
        try:
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                ),
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as error:
            print("Gemini API error:", error)  # the real error shows in your terminal
            # Tell the page what kind of problem it was, so it can explain kindly
            yield f"[[CHAT_ERROR:{classify_error(error)}]]"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


# ---------------------------------------------------------------------
# Chat history (the sidebar)
# ---------------------------------------------------------------------

class SavedMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)
    had_photo: bool = False


class ChatSave(BaseModel):
    messages: list[SavedMessage] = Field(min_length=1, max_length=200)


def make_title(messages: list[SavedMessage]) -> str:
    """The chat title is the start of your first message."""
    for m in messages:
        if m.role == "user":
            text = " ".join(m.content.split())
            if text:
                return text[:50]
    return "New chat"


def store_messages(db: sqlite3.Connection, chat_id: int, messages: list[SavedMessage]) -> None:
    """Replace all messages of a chat with the given list."""
    db.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
    db.executemany(
        "INSERT INTO chat_messages (chat_id, position, role, content, had_photo) "
        "VALUES (?, ?, ?, ?, ?)",
        [(chat_id, i, m.role, m.content, int(m.had_photo)) for i, m in enumerate(messages)],
    )


@app.get("/api/chats")
def list_chats():
    """The chats for the sidebar, most recently used first."""
    with closing(get_db()) as db:
        rows = db.execute(
            "SELECT id, title, updated_at FROM chats "
            "ORDER BY updated_at DESC, id DESC LIMIT 100"
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/chats")
def create_chat(item: ChatSave):
    """Save a brand new chat."""
    now = datetime.now().isoformat(timespec="seconds")
    title = make_title(item.messages)
    with closing(get_db()) as db:
        cursor = db.execute(
            "INSERT INTO chats (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        chat_id = cursor.lastrowid
        store_messages(db, chat_id, item.messages)
        db.commit()
    return {"id": chat_id, "title": title}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: int):
    """Load one chat with all its messages."""
    with closing(get_db()) as db:
        found = db.execute("SELECT id, title FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if found is None:
            raise HTTPException(status_code=404, detail="That chat was not found.")
        rows = db.execute(
            "SELECT role, content, had_photo FROM chat_messages "
            "WHERE chat_id = ? ORDER BY position",
            (chat_id,),
        ).fetchall()
    return {
        "id": found["id"],
        "title": found["title"],
        "messages": [
            {"role": r["role"], "content": r["content"], "had_photo": bool(r["had_photo"])}
            for r in rows
        ],
    }


@app.put("/api/chats/{chat_id}")
def update_chat(chat_id: int, item: ChatSave):
    """Save the latest messages of an existing chat."""
    now = datetime.now().isoformat(timespec="seconds")
    with closing(get_db()) as db:
        exists = db.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="That chat was not found.")
        store_messages(db, chat_id, item.messages)
        db.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
        db.commit()
    return {"id": chat_id}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int):
    """Delete one chat (its messages are deleted with it)."""
    with closing(get_db()) as db:
        db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        db.commit()
    return {"deleted": True}


# ---------------------------------------------------------------------
# Mistake log
# ---------------------------------------------------------------------

class ExtractMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class ExtractRequest(BaseModel):
    messages: list[ExtractMessage] = Field(max_length=60)


class MistakeDraft(BaseModel):
    """What Gemini fills in for the form. The student checks it before saving."""
    found_question: bool
    section: Section
    concept: str
    my_answer: str
    correct_answer: str
    note: str


class MistakeIn(BaseModel):
    """What the browser sends when you press Save."""
    section: Section
    concept: str = Field(max_length=100)
    my_answer: str = Field(default="", max_length=200)
    correct_answer: str = Field(default="", max_length=200)
    note: str = Field(default="", max_length=500)


@app.post("/api/mistakes/extract")
def extract_mistake(request: ExtractRequest):
    """Ask Gemini to fill in the 'Save this mistake' form from the chat."""
    transcript = "\n\n".join(
        ("Student: " if m.role == "user" else "Tutor: ") + m.content[:4000]
        for m in request.messages[-30:]
    )

    # Show Gemini the concept names you already use, so it reuses them
    with closing(get_db()) as db:
        rows = db.execute(
            "SELECT concept FROM mistakes GROUP BY concept COLLATE NOCASE "
            "ORDER BY MAX(id) DESC LIMIT 40"
        ).fetchall()
    known = [row["concept"] for row in rows]

    prompt = (
        "Existing concept names (reuse one if it matches): "
        + (", ".join(known) if known else "none yet")
        + "\n\nChat:\n"
        + transcript
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=EXTRACT_PROMPT,
                response_mime_type="application/json",
                response_schema=MistakeDraft,
                thinking_config=types.ThinkingConfig(thinking_level="low"),
            ),
        )
        return MistakeDraft.model_validate_json(response.text)
    except Exception as error:
        print("Extract error:", error)  # visible in your terminal
        raise HTTPException(
            status_code=502,
            detail="I couldn't fill in the form automatically.",
        )


@app.post("/api/mistakes")
def save_mistake(item: MistakeIn):
    """Save one mistake."""
    # Tidy the concept: remove extra spaces
    concept = " ".join(item.concept.split())
    if not concept:
        raise HTTPException(status_code=400, detail="Please type the concept.")

    with closing(get_db()) as db:
        # If you already saved this concept (any capital letters), reuse its spelling
        existing = db.execute(
            "SELECT concept FROM mistakes WHERE concept = ? COLLATE NOCASE LIMIT 1",
            (concept,),
        ).fetchone()
        if existing:
            concept = existing["concept"]

        # The ? marks keep your text safe: it is never mixed into the SQL itself
        cursor = db.execute(
            "INSERT INTO mistakes (date, section, concept, my_answer, correct_answer, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                date.today().isoformat(),
                item.section,
                concept,
                item.my_answer.strip(),
                item.correct_answer.strip(),
                item.note.strip(),
            ),
        )
        db.commit()
        row = db.execute("SELECT * FROM mistakes WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)


@app.get("/api/mistakes")
def list_mistakes(concept: str | None = None):
    """List saved mistakes, newest first. Add ?concept=... to filter."""
    sql = "SELECT * FROM mistakes"
    params = []
    if concept:
        sql += " WHERE concept = ? COLLATE NOCASE"
        params.append(concept)
    sql += " ORDER BY id DESC"

    with closing(get_db()) as db:
        rows = db.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/concepts")
def list_concepts():
    """Each concept with how many times you missed it, most missed first."""
    with closing(get_db()) as db:
        rows = db.execute(
            "SELECT concept, COUNT(*) AS count FROM mistakes "
            "GROUP BY concept COLLATE NOCASE "
            "ORDER BY count DESC, MAX(id) DESC"
        ).fetchall()
    return [dict(row) for row in rows]


@app.delete("/api/data")
def delete_all_data():
    """Delete ALL your data: every mistake and every chat. Then shrink the file."""
    with closing(get_db()) as db:
        cursor = db.execute("DELETE FROM mistakes")
        deleted = cursor.rowcount
        db.execute("DELETE FROM chats")  # the messages inside are deleted too
        db.execute(
            "DELETE FROM sqlite_sequence WHERE name IN ('mistakes', 'chats', 'chat_messages')"
        )
        db.commit()
        db.execute("VACUUM")
    return {"deleted": deleted}