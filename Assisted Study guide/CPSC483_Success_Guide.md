# How to Succeed in CPSC 483 (Intro to ML) — Summer 2026

> A 4-week sprint. ~1 textbook chapter per 3-hour session. The single biggest risk is falling behind by even one day. This guide is built to prevent that.

---

## The one thing that matters most
This course covers in **~4 weeks** what a normal semester spreads over 15. There is no recovery time. The students who do well aren't smarter — they **never let a session go un-consolidated**. Treat each lecture day as a closed loop: prime → attend → consolidate → run the code → recall. Do that loop daily and the course is very doable.

## Your weekly rhythm (Mon/Tue/Wed)
The course meets 9am–12pm. Build the day around that:

- **Night before (30–45 min):** Skim the assigned chapter. Don't master it — just collect the vocabulary and the 2–3 big ideas so the 3-hour lecture lands on prepared ground instead of cold. Ask me for a "pre-lecture primer" and I'll generate exactly this.
- **In lecture:** Take notes against the slides, not from scratch. Mark every place the slides *differ from or emphasize something the book glosses over* — instructor exams tend to follow the slides.
- **Same afternoon/evening (45–60 min):** Consolidate while it's fresh. Reconcile slides + textbook into a short summary and a few recall questions. This is the step people skip, and it's the step that decides the grade.
- **Run the chapter's Colab notebook by hand.** Don't just read it — retype key cells, change a parameter, break it, see what happens. The exams are AI-free and 50% of your grade; real fluency only comes from doing.

## The Colab / Jupyter workflow (your stated approach — let's optimize it)
Since you're working exercises in Jupyter on Google Colab:

- **Keep one notebook per chapter**, named `ch03_classification.ipynb`, etc. Mirror the textbook's structure with markdown headers so it doubles as searchable notes.
- **Annotate as you go.** Above each code block, write one markdown sentence: *what* it does and *why*. Future-you reviewing for the midterm will thank you.
- **Prediction-first habit:** before running a cell, write a one-line markdown comment predicting the output. Compare. The gap between your prediction and reality is exactly what you don't understand yet — flag those.
- **A "scratch" cell at the top** where you paste anything confusing. Bring those to me or to office hours.
- **Save/version:** if you use the CSUF VCL, sessions are wiped after ~4 hours — `git push` or download before you log off. Colab autosaves to Drive, which is safer for this.
- For homework code: the syllabus requires comments, descriptive names, and a header comment with your name/email/assignment. Set up a snippet so every notebook starts with that header.

## Front-load the deadline traps
These have lead times and will collide with finals if you wait:
- **Form your group (1–3 people)** in the first few days.
- **Pick a project dataset direction early** — presentations are **6/24**, the day before the final. Real-world dataset required.
- **Grad-only ~2-page report:** email Dr. Panangadan now for the suggested topic list, pick something genuinely interesting, draft it during a lighter week (not finals week). Sections required: problem, significance, solution, pros/cons, conclusion, references.

## Exam strategy (midterm 6/11, final 6/25)
- **Both exams are individual and AI-free.** Your prep target is *unaided* fluency, not recognition. When you study with me, have me quiz you (active recall) rather than explain at you.
- **Neither exam is cumulative.** The final covers *only* post-midterm material (Ch 8, 9, 10–11, 12, 19). Don't waste finals-week energy re-reviewing Ch 1–7.
- **Midterm covers Ch 1–7 + SVM.** Use the separate midterm study guide. Whether you can bring notes is instructor-dependent ("aids as described by the instructor") — confirm with Dr. Panangadan; if notes are allowed, the study guide is your sheet.
- Practice the end-of-chapter questions in Géron. They're a good proxy for the level expected.

## Use the support that's there
- **Office hours: Mon–Wed 4:45–5:45pm** (in person + Zoom), same days you have class. Email is the instructor's preferred contact; ~2 working-day response.
- **You** are encouraged to use AI for code and rewriting short answers (just be able to show your own contribution). That's where I come in — but always in "help me understand" mode, because the exams will expose anything you only pretended to learn.

## How to get the most out of our sessions
- Start each chat by re-uploading the latest `skills.md` so I'm instantly oriented.
- Tell me which mode you want: **primer**, **consolidate**, **explain a fuzzy concept**, **debug code**, **quiz me**, or **project help**.
- After each session I update `skills.md` (concept log + open items) so the next chat builds on this one.

---
*Built 2026-05-26. Tell me what worked for you in Adv DB and I'll personalize this further.*
