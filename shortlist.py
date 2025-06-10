import openai
from sentence_transformers import SentenceTransformer, util

# Set your OpenAI API Key
openai.api_key = "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"  # Replace with your key or use environment variable

# --------- INPUT FUNCTION ---------
def collect_inputs():
    print("\n📄 Enter the Job Description:")
    job_description = input("> ")

    print("\n📄 Paste 5 resume texts (press Enter after each one):")
    resumes = []
    for i in range(5):
        print(f"\nResume {i+1}:")
        resume = input("> ")
        resumes.append(resume)

    return resumes, job_description


# --------- GPT QUESTION GENERATOR ---------
def generate_gpt_questions(resume_text, job_description, resume_id):
    print(f"\n🤖 GPT Interview Questions for Resume {resume_id}:\n")

    prompt = f"""
You are an expert technical interviewer.

Given the following resume and job description, generate 5 intelligent, context-aware technical interview questions.

Resume:
\"\"\"
{resume_text}
\"\"\"

Job Description:
\"\"\"
{job_description}
\"\"\"

Questions:
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if needed
            messages=[
                {"role": "system", "content": "You are a technical interviewer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        questions = response['choices'][0]['message']['content']
        print(questions)

    except Exception as e:
        print(f"⚠️ Error generating questions: {e}")


# --------- BERT RANKING ---------
def rank_with_bert(resumes, jd):
    print("\n--- 🤖 BERT Ranking (SentenceTransformer) ---")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([jd] + resumes, convert_to_tensor=True)

    jd_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]

    scores = util.cos_sim(jd_embedding, resume_embeddings)[0]
    ranked = sorted([(i+1, resumes[i], float(scores[i])) for i in range(len(resumes))], key=lambda x: x[2], reverse=True)

    for rank, (idx, res, score) in enumerate(ranked, start=1):
        print(f"\n🏅 Rank {rank} | Resume {idx} | Score: {score:.2f}\n")

        if rank <= 3:
            generate_gpt_questions(resume_text=res, job_description=jd, resume_id=idx)


# --------- MAIN ---------
if __name__ == "__main__":
    resumes, job_description = collect_inputs()
    rank_with_bert(resumes, job_description)
