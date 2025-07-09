import 'package:flutter/material.dart';


class ReportPage extends StatefulWidget {
  const ReportPage({super.key});

  @override
  State<ReportPage> createState() => _ReportPageState();
}

class _ReportPageState extends State<ReportPage> {
  @override
  Widget build(BuildContext context) {
  return Scaffold(
      backgroundColor: Color.fromARGB(255, 249, 250, 251),
      
      body: Center(
       
                child: Text(
                  "Welcome Back Aman!\n",
                  textAlign: TextAlign.center,
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                    color: Color.fromARGB(255, 100, 92, 92),
                  ),
                ),
              ),
    );

}

}