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
                crossAxisAlignment: CrossAxisAlignment.center,
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
