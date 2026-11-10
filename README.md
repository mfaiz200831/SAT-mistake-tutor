# SAT Mistake Tutor

An AI tutor for the Digital SAT that helps me learn from my mistakes instead of just giving answers. I built it because [one sentence: your real reason].

## What it does
- Chat with an AI tutor that streams answers and remembers the conversation
- Reads a photo of an SAT question (JPEG from the camera) and shows what it read so I can correct it
- Finds the section and exact concept behind a mistake, using the 8 Digital SAT categories
- Asks what I chose first, then gives one hint at a time before the full solution
- Explains the fastest method, including Desmos for math, and why the wrong answer was tempting
- Writes 3 new practice questions (easy, medium, hard) on the same concept
- Chat history: saves chats in a sidebar, reopen or delete them
- Mistake log: "Save this mistake" fills in a form automatically (section, concept, my answer, correct answer, note), which I check before saving
- Honest feedback: it never flatters and says "I'm not sure" when it isn't

## Why it's different
Most chatbots just give answers. This one is built around finding why I got a question wrong and practicing that concept.

## Built with
Python, FastAPI, SQLite, HTML/CSS/JavaScript, Google Gemini API

## How it works
- The backend (main.py) holds the API key and talks to Gemini. The key never goes in the web page.
- A stronger Gemini model is used only when a photo is in the chat.
- Chats and mistakes are saved in a local SQLite file on my computer.
- Friendly error messages for free-limit reached, bad key, busy servers.

## How to run it
1. Install Python
2. Run `pip install -r requirements.txt`
3. Create a file named `.env` containing `GEMINI_API_KEY=your_key_here`
4. Run `uvicorn main:app --reload`
5. Open http://127.0.0.1:8000

## Why i built it
I built it because I was losing SAT points on the same concepts and wanted a tool that finds the gap behind each mistake instead of only giving answers. General chatbots solve the question for you, but I wanted one that makes me try first and then practice the weak concept.


## What I learned
- How a web page is put together: the page you see, the code behind it, and how they talk to each other
- How to connect an app to an AI service and make it answer the way I want
- How to save data, like chats and my SAT mistakes
- How to keep secret keys safe and put a project on GitHub
- How to test my work and fix problems

I built this with the help of AI tools, and I tested and improved it myself.

