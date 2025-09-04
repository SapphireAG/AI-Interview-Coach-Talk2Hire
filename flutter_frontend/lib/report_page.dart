import 'package:flutter/material.dart';
import 'dart:async';

class ReportPage extends StatefulWidget {
  final String username;
  const ReportPage({super.key, required this.username});

  @override
  State<ReportPage> createState() => _ReportPageState();
}

class _ReportPageState extends State<ReportPage> {
  final PageController _pageController = PageController(viewportFraction: 0.85);
  int _currentPage = 0;
  late Timer _autoScrollTimer;

  final List<String> features = [
    'Technical Questions',
    'Personalized Questions',
    'Scenario-Based Questions',
  ];

  final Map<String, String> featureIcons = {
    'Technical Questions': 'assets/tech.png',
    'Personal Questions': 'assets/personal.png',
    'Scenario-Based Questions': 'assets/scenario.png',
  };

  @override
  void initState() {
    super.initState();
    _autoScrollTimer = Timer.periodic(Duration(seconds: 4), (timer) {
      if (_pageController.hasClients) {
        _currentPage = (_currentPage + 1) % features.length;
        _pageController.animateToPage(
          _currentPage,
          duration: Duration(milliseconds: 500),
          curve: Curves.easeInOut,
        );
      }
    });
  }

  @override
  void dispose() {
    _pageController.dispose();
    _autoScrollTimer.cancel();
    super.dispose();
  }

  

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color.fromARGB(255, 249, 250, 251),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
          child: Column(
            children: [
              // Top Row: Menu icon, Welcome Text, Profile icon
              Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  // Menu Icon (Top Left)
                  IconButton(
                    icon: const Icon(
                      Icons.arrow_back_ios,
                      size: 25,
                      color: Color.fromARGB(255, 100, 92, 92),
                    ),
                    onPressed: () {
                      Navigator.pushNamed(context, '/questions_screen', arguments: widget.username);
                    },
                  ),

                  // Welcome Text (Top Center)
                  Expanded(
                    child: Align(
                      alignment: Alignment.topCenter,
                      child: Text(
                        "Catch Up on Your Progress",
                        style: const TextStyle(
                          fontSize: 25,
                          fontWeight: FontWeight.bold,
                          color: Color.fromARGB(255, 100, 92, 92),
                        ),
                      ),
                    ),
                  ),

                  // Profile Icon (Top Right)
                  const Icon(
                    Icons.account_circle,
                    size: 40,
                    color: Color.fromARGB(255, 100, 92, 92),
                  ),
                ],
              ),

              const SizedBox(height: 6),

              // Streak Text aligned to right below profile icon
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: const [
                  Icon(
                    Icons.local_fire_department,
                    color: Colors.orange,
                    size: 22,
                  ),
                  SizedBox(width: 4),
                  Text(
                    "3 days",
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: Color.fromARGB(255, 100, 92, 92),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 30),

              // Fixed 2x2 stat grid using Rows
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8.0),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _buildStatCard("Questions Attempted\nThis Week"),
                        _buildStatCard("Top Performing\nArea"),
                      ],
                    ),
                    const SizedBox(height: 20),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        _buildStatCard("Your Weekly\nHigh Score"),
                        _buildStatCard("Dominant Emotion\nThis Week"),
                      ],
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 40),

              // Auto-scroll carousel title (optional)
              Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  "TODAY'S ACTIVITY",
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    color: Color.fromARGB(255, 90, 60, 9),
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // Auto-scroll carousel
              SizedBox(
                height: 150,
                child: PageView.builder(
                  controller: _pageController,
                  itemCount: features.length,
                  itemBuilder: (context, index) {
                    return Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 8.0),
                      child: Container(
                        decoration: BoxDecoration(
                          color: Color.fromARGB(255, 123, 207, 222),
                          borderRadius: BorderRadius.circular(16),
                          boxShadow: [
                            BoxShadow(color: Colors.black12, blurRadius: 6),
                          ],
                        ),
                        padding: const EdgeInsets.all(16),
                      child: Column(
  crossAxisAlignment: CrossAxisAlignment.start,
  children: [
    Row(
      children: [
        Image.asset(
          featureIcons[features[index]] ?? '',
          height: 40,
          width: 40,
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            features[index],
            style: const TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Color.fromARGB(255, 90, 60, 9),
            ),
          ),
        ),
      ],
    ),
    // Optional: add extra space below if needed
  ],
),

                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatCard(String label) {
    return Container(
      width: 150,
      height: 150,
      padding: const EdgeInsets.all(12.0),
      decoration: BoxDecoration(
        color: Color.fromARGB(255, 237, 237, 237),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Align(
        alignment: Alignment.topLeft,
        child: Text(
          label,
          textAlign: TextAlign.start,
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.bold,
            color: Color.fromARGB(255, 90, 60, 9),
          ),
        ),
      ),
    );
  }
}