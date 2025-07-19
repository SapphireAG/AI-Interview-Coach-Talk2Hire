import 'package:flutter/material.dart';


class FeedbackPage extends StatefulWidget {
  
  const FeedbackPage({super.key});

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  // These will be replaced by LLM-generated feedback
  String performanceText =
      "Your answer was clear and confident, but you hesitated a little in parts. Ending with a strong conclusion will make your point more memorable.";

  String improvementText =
      "Try adding a real-world example to make your answer more compelling. A brief summary at the end will reinforce your key message.";

  // TO BE REPLACED WITH LLM GENERATED FEEDBACK LATER

  //   setState(() {
  //   performanceText = llmResponse["performance"]; 
  //   improvementText = llmResponse["improvement"]; 
  // }); 

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color.fromARGB(255, 249, 250, 251),
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
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                    color: Color.fromARGB(255, 100, 92, 92),
                  ),
                ),
              ),
              const SizedBox(height: 30),

              // Question number tag
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // Back Arrow (left end)
                  IconButton(
                    icon: const Icon(Icons.arrow_back),
                    onPressed: () {
                      // Handle back navigation
                    },
                  ),

                  // Question Label (center)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: const Color.fromARGB(255, 237, 237, 237),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Text(
                      "Question 3",
                      style: TextStyle(
                        fontSize: 25,
                        color: Color.fromARGB(255, 90, 60, 9),
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),

                  // Forward Arrow (right end)
                  IconButton(
                    icon: const Icon(Icons.arrow_forward),
                    onPressed: () {
                      // Handle forward navigation
                    },
                  ),
                ],
              ),

              const SizedBox(height: 30),

              // Performance Breakdown
              const Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  "Performance Breakdown",
                  style: TextStyle(
                    fontSize: 25,
                    fontWeight: FontWeight.bold,
                    color: Color.fromARGB(255, 100, 92, 92),
                  ),
                ),
              ),
              const SizedBox(height: 10),
              _buildFeedbackCard(performanceText),

              const SizedBox(height: 30),

              // Suggested Improvements
              const Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  "Suggested Improvements",
                  style: TextStyle(
                    fontSize: 25,
                    fontWeight: FontWeight.bold,
                    color: Color.fromARGB(255, 100, 92, 92),
                  ),
                ),
              ),
              const SizedBox(height: 10),
              _buildFeedbackCard(improvementText),
            ],
          ),
        ),
      ),
    );
  }

  // Feedback Card Builder Function
  Widget _buildFeedbackCard(String message) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color.fromARGB(255, 237, 237, 237),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.question_mark_rounded, color: Colors.black),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(
                color: Color.fromARGB(255, 100, 92, 92),
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
