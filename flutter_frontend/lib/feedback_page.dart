import 'package:flutter/material.dart';

class FeedbackPage extends StatefulWidget {
  const FeedbackPage({super.key});
  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  @override
  Widget build(BuildContext context) {
     return Scaffold(
      backgroundColor: Color.fromARGB(255, 249, 250, 251),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 32),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              const Center(
                child: Text( 
                  "Question-Wise\nFeedback Insights",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 26,
                    fontWeight: FontWeight.bold,
                    color: Color.fromARGB(255, 100 , 92 , 92),
                  ),
                ),
              ),
              const SizedBox(height: 20),

              // Question number tag
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: Color.fromARGB(255, 237, 237, 237),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Text(
                  "Question 3",
                  style: TextStyle(
                    fontSize: 18,
                    color: Color.fromARGB(255, 90, 60, 9),
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              const SizedBox(height: 30),

              // Performance Breakdown
              const Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  "Performance Breakdown",
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(height: 10),
              _feedbackCard(
                "Your answer was clear and confident, but you hesitated a little in parts. Ending with a strong conclusion will make your point more memorable.",
              ),
              const SizedBox(height: 30),

              // Suggested Improvements
              const Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  "Suggested Improvements",
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(height: 10),
              _feedbackCard(
                "Try adding a real-world example to make your answer more compelling. A brief summary at the end will reinforce your key message.",
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _feedbackCard(String message) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Color(0xFF7A87F9),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.question_mark_rounded, color: Colors.white),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 15,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}