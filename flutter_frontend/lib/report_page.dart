import 'package:flutter/material.dart';
import 'dart:async';

class ReportPage extends StatefulWidget {
  final String username;
  const ReportPage({super.key, required this.username});

  @override
  State<ReportPage> createState() => _ReportPageState();
}

class _ReportPageState extends State<ReportPage> {
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
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Menu Icon (Top Left)
                  const Icon(
                    Icons.arrow_back_ios,
                    size: 25,
                    color: Color.fromARGB(255, 100, 92, 92),
                  ),

                  // Welcome Text (Top Center)
                  Expanded(
                    child: Align(
                      alignment: Alignment.topCenter,
                      child: Text(
                        "Welcome Back ${widget.username}!",
                        style: const TextStyle(
                          fontSize: 30,
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
                    "3-day streak",
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: Color.fromARGB(255, 100, 92, 92),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 30),

              // Grid of 4 stat cards
              Expanded(
                child: GridView.count(
                  crossAxisCount: 2,
                  crossAxisSpacing: 16,
                  mainAxisSpacing: 16,
                  childAspectRatio: 1,
                  children: [
                    _buildStatCard("Questions Attempted\nThis Week"),
                    _buildStatCard("Top Performing\nArea"),
                    _buildStatCard("Your Weekly\nHigh Score"),
                    _buildStatCard("Dominant Emotion\nThis Week"),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStatCard(String title) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.15),
            spreadRadius: 1,
            blurRadius: 8,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Align(
        alignment: Alignment.topLeft,
        child: Text(
          title,
          style: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 16,
            color: Color.fromARGB(255, 100, 92, 92),
          ),
        ),
      ),
    );
  }
}
