from django.db import models

class MentalHealthResponse(models.Model):
    question_id = models.CharField(max_length=100, unique=True)
    question_title = models.TextField()
    question_text = models.TextField()
    topic = models.CharField(max_length=255)
    therapist_info = models.TextField()
    answer_text = models.TextField()
    upvotes = models.IntegerField()

    def __str__(self):
        return self.question_title
