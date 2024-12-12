import os
import openai
import fitz
import re
import nltk
from flask import Flask, render_template, request, session, jsonify
import logging
from dotenv import load_dotenv
import uuid
import requests
from werkzeug.utils import secure_filename
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apscheduler.schedulers.background import BackgroundScheduler  # For scheduling the daily email
from apscheduler.triggers.cron import CronTrigger  # For scheduling at a specific time

load_dotenv()

app = Flask(__name__)

app.secret_key = os.getenv('SECRET_KEY')

# Set up OpenAI and xAI API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
openai_organization = os.getenv('OPENAI_ORGANIZATION')
xai_api_key = os.getenv('XAI_API_KEY')

logging.basicConfig(level=logging.INFO)

# SMTP Configuration
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USERNAME = os.getenv('SMTP_USERNAME')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
EMAIL_FROM = os.getenv('EMAIL_FROM', SMTP_USERNAME)

# Email recipient and send time (set via environment variables)
DAILY_EMAIL_RECIPIENT = os.getenv('DAILY_EMAIL_RECIPIENT')
DAILY_EMAIL_TIME = os.getenv('DAILY_EMAIL_TIME')  # Format: 'HH:MM' in 24-hour time

# Scheduler
scheduler = BackgroundScheduler()
scheduler.start()

def clean_text(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = re.sub(r' \.', '.', cleaned_text)
    cleaned_text = re.sub(r' ,', ',', cleaned_text)
    cleaned_text = re.sub(r' !', '!', cleaned_text)
    cleaned_text = re.sub(r' \?', '?', cleaned_text)
    cleaned_text = re.sub(r' ;', ';', cleaned_text)
    cleaned_text = re.sub(r' :', ':', cleaned_text)
    return cleaned_text

def extract_pdf_text(pdf_data):
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    sentences = nltk.sent_tokenize(text)
    paragraphs = [' '.join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]
    text = '\n\n'.join(paragraphs)
    text = text.replace(' .', '.')
    return clean_text(text)

def parse_discussion(discussion_text):
    # Ensure consistent line breaks
    discussion_text = discussion_text.strip()
    # Replace multiple spaces or tabs with a single space
    discussion_text = re.sub(r'[ \t]+', ' ', discussion_text)
    # Replace any \r\n or \r with \n
    discussion_text = discussion_text.replace('\r\n', '\n').replace('\r', '\n')
    # Insert a newline before each speaker if missing
    discussion_text = re.sub(r'(?<!\n)(Person [ABC]|Teacher):', r'\n\1:', discussion_text)
    # Split the text into lines
    lines = discussion_text.strip().split('\n')
    discussion_lines = []
    for line in lines:
        match = re.match(r'(Person [ABC]|Teacher):\s*(.*)', line)
        if match:
            speaker = match.group(1)
            text = match.group(2).strip()
            discussion_lines.append({'speaker': speaker, 'text': text})
    return discussion_lines

def generate_speech(text, voice):
    response = requests.post(
        'https://api.openai.com/v1/audio/speech',
        headers={
            'Authorization': f'Bearer {openai.api_key}',
            'Content-Type': 'application/json'
        },
        json={
            'input': text,
            'voice': voice,
            'model': 'tts-1'
        },
    )
    if response.status_code == 200:
        return response.content  # Audio content in binary format
    else:
        raise Exception(f"TTS API request failed: {response.text}")

def send_email(recipient_email, subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, recipient_email, msg.as_string())
        logging.info(f"Email sent to {recipient_email}")
    except Exception as e:
        logging.error(f"Error sending email to {recipient_email}: {e}")

def daily_email_task():
    recipient_email = DAILY_EMAIL_RECIPIENT
    if not recipient_email:
        logging.warning("DAILY_EMAIL_RECIPIENT not set. Skipping daily email task.")
        return

    # Collect all discussion files
    discussions_dir = 'discussions'
    if not os.path.exists(discussions_dir):
        logging.warning("No discussions directory found. Skipping daily email task.")
        return

    discussion_files = []
    for root, dirs, files in os.walk(discussions_dir):
        for file in files:
            if file.endswith('.txt'):
                discussion_files.append(os.path.join(root, file))

    if not discussion_files:
        logging.warning("No discussion files found. Skipping daily email task.")
        return

    import random
    # Select a random discussion file
    discussion_file = random.choice(discussion_files)
    with open(discussion_file, 'r', encoding='utf-8') as f:
        discussion_text = f.read()
    # Send the email
    subject = "Your Daily Discussion Reminder"
    body = f"Hello,\n\nHere is a discussion from your PDF Book Club:\n\n{discussion_text}\n\nBest regards,\nPDF Book Club"
    send_email(recipient_email, subject, body)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        pdf_data = pdf_file.read()
        pdf_filename = pdf_file.filename  # Get the uploaded file's name
        # Store the sanitized PDF file name in the session
        session['pdf_filename'] = secure_filename(pdf_filename)

        text = extract_pdf_text(pdf_data)
        tokens = nltk.word_tokenize(text)
        segments = [' '.join(tokens[i:i+2000]) for i in range(0, len(tokens), 2000)]
        segments = [clean_text(segment) for segment in segments]
        enumerated_segments = list(enumerate(segments))
        session['segments'] = segments  # Store segments in session
        return render_template('result.html', segments=enumerated_segments, discussions=session.get('discussions'))
    session.pop('discussions', None)
    session.pop('pdf_filename', None)
    session.pop('segments', None)
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    index = request.form.get('index')
    segment = request.form.get('segment')
    relation_text = request.form.get('relation_text')
    additional_turn = request.form.get('additional_turn', 'false')

    if additional_turn == 'true':
        # Get previous discussions from session
        previous_discussions = session.get('discussions', {}).get(str(index), [])
        previous_discussion_texts = [disc['text'] for disc in previous_discussions]
        previous_discussion = '\n\n'.join(previous_discussion_texts)

        # Prepare the prompt for additional turn with Teacher
        prompt = f"""
        Continue the following book club discussion between three people about the following section of a book:

        {segment}

        Previous discussion:

        {previous_discussion}

        Now, a Teacher joins the discussion. The Teacher's role is to challenge the participants to be more specific in drawing from the text, encouraging them to make cross comparisons between fragments within the text and the full passage. If the participants are loosely summarizing based on simple opinions, the Teacher should guide them to delve deeper.

        Please continue the discussion, including the Teacher and the three participants. The discussion should be formatted as follows:

        Person A: [Person A's comment]

        Person B: [Person B's comment]

        Person C: [Person C's comment]

        Teacher: [Teacher's comment]

        Ensure that each speaker's comment starts on a new line, with a blank line separating each speaker's comment.
        """

        if relation_text:
            prompt += f" Next, have the group participants relate the reading to {relation_text}."
    else:
        # Original prompt
        prompt = f"""
        Can you generate for me a short book club discussion between three people about the following section of a book:

        {segment}

        The discussion should be formatted as follows:

        Person A: [Person A's comment]

        Person B: [Person B's comment]

        Person C: [Person C's comment]

        Ensure that each speaker's comment starts on a new line, with a blank line separating each speaker's comment.

        This discussion should involve at least two "turns" per discussion participant in this discussion. The discussion should address the book club question: 'What is a random word or phrase that stood out to you in reading this text? What does that word or phrase bring to mind for you? Then, relate your thought back to the passage's message.'
        """

        if relation_text:
            prompt += f" Next, have the group participants relate the reading to {relation_text}."

    # Save the original api_base and api_key
    original_api_base = openai.api_base
    original_api_key = openai.api_key

    # Set xAI API base and API key for chat completion
    openai.api_base = 'https://api.x.ai/v1'
    openai.api_key = xai_api_key

    try:
        response = openai.ChatCompletion.create(
            model="grok-beta",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        # Restore original API settings before raising exception
        openai.api_base = original_api_base
        openai.api_key = original_api_key
        raise e

    # Restore original API settings for TTS
    openai.api_base = original_api_base
    openai.api_key = original_api_key

    discussion = response.choices[0].message['content']
    logging.info(f"Raw discussion output: {discussion}")

    # Parse the discussion
    discussion_lines = parse_discussion(discussion)

    # Store the discussion in session without audio
    if 'discussions' not in session:
        session['discussions'] = {}
    if str(index) not in session['discussions']:
        session['discussions'][str(index)] = []
    session['discussions'][str(index)].append({
        'text': discussion,
        'lines': discussion_lines,
        'audio_generated': False  # Flag to indicate audio is not generated yet
    })
    session.modified = True  # To ensure the session is saved
    logging.info(f"Sessions after generation:  {session['discussions']}")

    # Save the discussion to a file
    pdf_filename = session.get('pdf_filename', 'Unknown_PDF')
    pdf_filename_sanitized = secure_filename(pdf_filename)
    discussion_num = len(session['discussions'][str(index)]) - 1  # Get the current discussion number
    # Construct the filename
    filename = f"Segment_{index}_Discussion_{discussion_num}_{pdf_filename_sanitized}.txt"
    discussions_dir = 'discussions'
    os.makedirs(discussions_dir, exist_ok=True)  # Ensure the directory exists
    file_path = os.path.join(discussions_dir, filename)
    # Save the discussion content to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(discussion)
    logging.info(f"Discussion saved to file: {file_path}")

    return jsonify(discussion=discussion, discussion_lines=discussion_lines)

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    index = request.form.get('index')
    discussion_num = int(request.form.get('discussion_num'))  # To identify which discussion to generate audio for

    # Retrieve the discussion from the session
    discussions = session.get('discussions', {}).get(str(index), [])
    if not discussions or discussion_num >= len(discussions):
        return jsonify({'error': 'Discussion not found'}), 404

    discussion_data = discussions[discussion_num]

    if discussion_data.get('audio_generated', False):
        return jsonify({'error': 'Audio already generated'}), 400

    discussion_lines = discussion_data['lines']

    # Map speakers to voices
    voice_mapping = {
        'Person A': 'shimmer',
        'Person B': 'onyx',
        'Person C': 'echo',
        'Teacher': 'nova'
    }

    # Generate audio for each line
    for line in discussion_lines:
        speaker = line['speaker']
        text = line['text']
        voice = voice_mapping.get(speaker, 'alloy')  # Default to 'alloy'
        try:
            # Generate audio
            audio_content = generate_speech(text, voice)
            # Generate unique filename
            audio_filename = f"{uuid.uuid4()}.mp3"
            audio_dir = os.path.join('static', 'audio')
            os.makedirs(audio_dir, exist_ok=True)  # Ensure directory exists
            audio_filepath = os.path.join(audio_dir, audio_filename)
            # Save audio file
            with open(audio_filepath, 'wb') as f:
                f.write(audio_content)
            # Store the audio file URL
            audio_url = f"/static/audio/{audio_filename}"
            # Add to line data
            line['audio_url'] = audio_url
        except Exception as e:
            logging.error(f"Error generating speech for {speaker}: {e}")
            line['audio_url'] = None  # Indicate that audio is not available

    # Update the discussion in the session
    discussion_data['lines'] = discussion_lines
    discussion_data['audio_generated'] = True
    session['discussions'][str(index)][discussion_num] = discussion_data
    session.modified = True

    # Return the updated discussion lines with audio URLs
    return jsonify({'discussion_lines': discussion_lines})

@app.route('/generate_segment_audio', methods=['POST'])
def generate_segment_audio():
    segment_text = request.form.get('segment_text')
    if not segment_text:
        return jsonify({'error': 'No segment text provided'}), 400

    # Define the maximum length per chunk (slightly less than 4096)
    MAX_CHARS = 4000

    # Split the text into chunks
    text_chunks = [segment_text[i:i+MAX_CHARS] for i in range(0, len(segment_text), MAX_CHARS)]

    audio_urls = []

    try:
        # Generate audio for each chunk
        for chunk in text_chunks:
            # Choose a valid voice
            voice = 'onyx'  # Valid voices: 'nova', 'shimmer', 'echo', 'onyx', 'fable', 'alloy'
            audio_content = generate_speech(chunk, voice)
            # Generate unique filename
            audio_filename = f"{uuid.uuid4()}.mp3"
            audio_dir = os.path.join('static', 'audio')
            os.makedirs(audio_dir, exist_ok=True)  # Ensure directory exists
            audio_filepath = os.path.join(audio_dir, audio_filename)
            # Save audio file
            with open(audio_filepath, 'wb') as f:
                f.write(audio_content)
            # Generate audio URL
            audio_url = f"/static/audio/{audio_filename}"
            audio_urls.append(audio_url)

        return jsonify({'audio_urls': audio_urls})
    except Exception as e:
        logging.error(f"Error generating speech for segment: {e}")
        return jsonify({'error': 'Failed to generate audio for the segment.'}), 500

@app.route('/format_rough', methods=['POST'])
def format_text_rough():
    segment = request.form.get('segment')

    # Import NLTK and download 'punkt' if necessary
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Use NLTK to split the segment into sentences
    sentences = nltk.sent_tokenize(segment)

    paragraphs = []
    paragraph = ''
    for i, sentence in enumerate(sentences):
        paragraph += sentence.strip() + ' '
        # Group sentences into paragraphs of 3 sentences each
        if (i + 1) % 3 == 0 or i == len(sentences) - 1:
            paragraphs.append(paragraph.strip())
            paragraph = ''

    logging.info(f"Type of paragraphs: {type(paragraphs)}")
    logging.info(f"Rough formatted text output: {paragraphs}")
    return jsonify({'formatted_text': paragraphs})

if __name__ == "__main__":
    # Schedule the daily email task at the specified time
    if DAILY_EMAIL_TIME:
        # Parse the time
        try:
            hour, minute = map(int, DAILY_EMAIL_TIME.split(':'))
            # Use CronTrigger to schedule at specific time every day
            scheduler.add_job(daily_email_task, CronTrigger(hour=hour, minute=minute), id='daily_email_task')
            logging.info(f"Scheduled daily email task at {DAILY_EMAIL_TIME}")
        except ValueError:
            logging.error("Invalid DAILY_EMAIL_TIME format. It should be in 'HH:MM' 24-hour format.")
    else:
        logging.warning("DAILY_EMAIL_TIME not set. Daily email task will not be scheduled.")

    app.run(host='0.0.0.0', port=4000)
