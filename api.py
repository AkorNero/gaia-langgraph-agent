import os

import requests
from datasets import load_dataset
from dotenv import load_dotenv

# from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field

load_dotenv()

token = os.getenv("HF_TOKEN")
api_endpoint: str = str(os.getenv("HF_ASSESMENT_API_ENDPOINT"))


class Question(BaseModel):
  task_id: str
  question: str = Field(alias="Question")
  level: str = Field(alias="Level")
  file_name: str  # empty string when there's no attachment
  file_path: str  # empty string when there's no attachment
  final_answer: str = Field(alias="Final answer")
  annotator_metadata: dict[str, str] = Field(alias="Annotator Metadata")


# class Answer(BaseModel):
#   task_id: str
#   submitted_answer: str

# class Submission(BaseModel):
#   username: str
#   agent_code: str                # public repo / Space URL
#   answers: list[Answer]

# class SubmissionResult(BaseModel):
#   username: str
#   score: float
#   correct_count: int
#   total_attempted: int
#   message: str


def get_questions() -> list[Question]:
  res = requests.get(f"{api_endpoint}/questions", timeout=30)
  res.raise_for_status()

  task_ids = {q["task_id"] for q in res.json()}

  # data_dir = snapshot_download(
  #   repo_id="gaia-benchmark/GAIA",
  #   repo_type="dataset",
  #   token=token,
  # )

  ds = load_dataset(
    "gaia-benchmark/GAIA",
    "2023_level1",
    split="validation",
    token=token,
  )

  with_files = ds.filter(lambda r: r["task_id"] in task_ids)

  with_files.to_csv("data.csv")

  return [Question(**q) for q in with_files.to_list()]
