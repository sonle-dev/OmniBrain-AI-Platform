from unittest.mock import patch, AsyncMock
from app.api.v1.quiz import _build_quiz_prompt, _parse_quiz_json
from app.schemas.requests import QuizGenerateRequest

def test_quiz_prompt_builder():
    """Test quiz prompt construction for Vietnamese multiple choice."""
    req = QuizGenerateRequest(
        topic="Lập trình Python",
        count=3,
        difficulty="medium",
        question_type="multiple_choice",
        language="vi"
    )
    prompt = _build_quiz_prompt(req)
    assert "Respond entirely in Vietnamese." in prompt
    assert 'Generate exactly 3 medium difficulty multiple-choice with 4 options (A, B, C, D) questions about: "Lập trình Python".' in prompt

def test_quiz_json_parser():
    """Test extracting structured QuizQuestions from raw LLM output with code fences."""
    raw_llm = """```json
    {
      "questions": [
        {
          "id": 1,
          "question": "Python là gì?",
          "options": [
            {"label": "A", "text": "Ngôn ngữ lập trình bậc cao"},
            {"label": "B", "text": "Hệ điều hành"},
            {"label": "C", "text": "Trình duyệt web"},
            {"label": "D", "text": "Phần cứng máy tính"}
          ],
          "correct_answer": "A",
          "explanation": "Python là ngôn ngữ lập trình đa năng bậc cao.",
          "difficulty": "medium"
        }
      ]
    }
    ```"""
    req = QuizGenerateRequest(topic="Python")
    questions = _parse_quiz_json(raw_llm, req)
    assert len(questions) == 1
    assert questions[0].question == "Python là gì?"
    assert questions[0].correct_answer == "A"
    assert len(questions[0].options) == 4

def test_quiz_generate_endpoint(client):
    """Test /api/v1/quiz/generate endpoint returning structured quiz questions."""
    mock_questions = [
        {
            "id": 1,
            "question": "Biến trong Python khai báo thế nào?",
            "options": [{"label": "A", "text": "x = 5"}, {"label": "B", "text": "int x = 5"}, {"label": "C", "text": "var x = 5"}, {"label": "D", "text": "dim x = 5"}],
            "correct_answer": "A",
            "explanation": "Python tự động định kiểu.",
            "difficulty": "easy"
        }
    ]
    mock_completion_res = {
        "text": "mock json",
        "model": "ollama/deepseek-r1:8b",
        "usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}
    }
    
    with patch("app.services.llm_factory.LLMFactory.chat_completion_with_self_correction", new_callable=AsyncMock) as mock_quiz:
        from app.schemas.responses import QuizQuestion, QuizOption
        parsed_q = [
            QuizQuestion(
                id=1,
                question=mock_questions[0]["question"],
                options=[QuizOption(**o) for o in mock_questions[0]["options"]],
                correct_answer="A",
                explanation=mock_questions[0]["explanation"],
                difficulty="easy"
            )
        ]
        mock_quiz.return_value = (parsed_q, mock_completion_res)
        
        response = client.post("/api/v1/quiz/generate", json={
            "topic": "Python Cơ Bản",
            "count": 1,
            "difficulty": "easy",
            "language": "vi"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["topic"] == "Python Cơ Bản"
        assert data["count"] == 1
        assert len(data["questions"]) == 1
        assert data["questions"][0]["correct_answer"] == "A"
