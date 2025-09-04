import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;

class QuestionsPage extends StatefulWidget {
  final String username;
  const QuestionsPage({super.key, required this.username});
  @override
  State<QuestionsPage> createState() => _QuestionsPageState();
}

String selectedCategory = "Scenario-Based Questions";
int _selectedValue = 10; // Default selection

void _startMock() {
  print("Mock started");
}

class _QuestionsPageState extends State<QuestionsPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        toolbarHeight: 100,
        title: Padding(
          padding: const EdgeInsets.only(top: 21.0),
          child: Center(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10.0),
              child: Image.asset(
                'assets/NEW_logo.png',
                height: 80.0,
                width: 150.0,
                fit: BoxFit.contain,
              ),
            ),
          ),
        ),
        backgroundColor: Color.fromARGB(255, 249, 250, 251),
        leading: Builder(
          builder: (context) => IconButton(
            icon: Icon(Icons.menu, color: Colors.black38, size: 35),
            onPressed: () {
              Scaffold.of(context).openDrawer();
            },
          ),
        ),
        actions: [
          IconButton(
            icon: Icon(Icons.account_circle),
            iconSize: 40,
            color: Colors.black38,
            onPressed: () {},
          ),
        ],
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            Container(
              height: 100,
              decoration: BoxDecoration(
                color: Color.fromARGB(255, 190, 233, 236),
              ),
              alignment: Alignment.centerLeft,
              padding: EdgeInsets.symmetric(horizontal: 16),
              child: Text(
                'Menu',
                style: TextStyle(
                  color: Color.fromARGB(255, 100, 92, 92),
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            ListTile(
              leading: Icon(Icons.analytics),
              title: Text('Performance Dashboard'),
              onTap: () {
                Navigator.pop(context);
                Navigator.pushNamed(context, '/report_page');
              },
            ),
          ],
        ),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.only(bottom: 120),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                height: 400.0,
                width: 400.0,
                padding: EdgeInsets.all(20.0),
                decoration: BoxDecoration(
                  color: Color.fromARGB(255, 237, 237, 237),
                  borderRadius: BorderRadius.circular(20.0),
                  boxShadow: [
                    BoxShadow( 
                      color: Colors.black12,
                      blurRadius: 4,
                      offset: Offset(2, 2),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    SizedBox(height: 20),
                    Container(
                      height: 50.0,
                      width: double.infinity,
                      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: DropdownButtonHideUnderline(
                        child: DropdownButton<String>(
                          borderRadius: BorderRadius.circular(12),
                          alignment: Alignment.center,
                          value: selectedCategory,
                          dropdownColor: Colors.white,
                          icon: Icon(Icons.arrow_drop_down, color: Colors.black),
                          items: [
                            DropdownMenuItem(
                              alignment: Alignment.center,
                              value: "Scenario-Based Questions",
                              child: Row(
                                children: [
                                  Icon(Icons.chat_bubble, size: 28),
                                  Text(
                                    "     Scenario Based Questions",
                                    textAlign: TextAlign.center,
                                    style: TextStyle(color: Colors.black),
                                  ),
                                ],
                              ),
                            ),
                            DropdownMenuItem(
                              alignment: Alignment.center,
                              value: "Technical Questions",
                              child: Row(
                                children: [
                                  Icon(Icons.computer, size: 28),
                                  Text(
                                    "     Technical Questions",
                                    textAlign: TextAlign.center,
                                    style: TextStyle(color: Colors.black),
                                  ),
                                ],
                              ),
                            ),
                            DropdownMenuItem(
                              alignment: Alignment.center,
                              value: "Personal Questions",
                              child: Row(
                                children: [
                                  Icon(Icons.person, size: 28),
                                  Text(
                                    "     Personal Questions",
                                    textAlign: TextAlign.center,
                                    style: TextStyle(color: Colors.black),
                                  ),
                                ],
                              ),
                            ),
                          ],
                          onChanged: (String? newValue) {
                            setState(() {
                              selectedCategory = newValue!;
                            });
                          },
                        ),
                      ),
                    ),
                    SizedBox(height: 30),
                    Row(
                      children: [
                        Icon(Icons.question_answer_outlined, color: Colors.black54),
                        Text(
                          '   Select the number of Questions',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            color: Color.fromARGB(255, 90, 60, 9),
                          ),
                        ),
                      ],
                    ),
                    SizedBox(height: 20),
                    _buildRadioOption(5, "5 Questions"),
                    SizedBox(height: 10),
                    _buildRadioOption(10, "10 Questions"),
                    SizedBox(height: 10),
                    _buildRadioOption(15, "15 Questions"),
                    SizedBox(height: 10),
                  ],
                ),
              ),
              SizedBox(height: 20),
              ElevatedButton(
                // onPressed: () async {
                //   _startMock();
                //   final response = await http.post(
                //     Uri.parse("http://127.0.0.1:8000/get-questions"),
                //     headers: {"Content-Type": "application/json"},
                //     body: jsonEncode({
                //       "question_type": selectedCategory.contains("Scenario")
                //           ? "Scenario"
                //           : selectedCategory.contains("Technical")
                //               ? "Technical"
                //               : "Behavioral",
                //       "count": _selectedValue,
                //       "score": 8,
                //     }),
                //   );
                //   if (response.statusCode == 200) {
                //     final jsonResp = jsonDecode(response.body);
                //     final List<dynamic> questions = jsonResp["questions"];
                //     Navigator.pushNamed(
                //       context,
                //       '/questions_page',
                //       arguments: questions,
                //     );
                //   } else {
                //     print("Failed to fetch questions: ${response.statusCode}");
                //     ScaffoldMessenger.of(context).showSnackBar(
                //       const SnackBar(content: Text("Failed to load questions")),
                //     );
                //   }
                // },


                // In the 'Start Mock Interview' ElevatedButton's onPressed
onPressed: () async {
  _startMock();
  // ... (http post request)
  final response = await http.post(
                    Uri.parse("http://127.0.0.1:8000/get-questions/"),
                    headers: {"Content-Type": "application/json"},
                    body: jsonEncode({
                      "question_type": selectedCategory.contains("Scenario")
                          ? "Scenario"
                          : selectedCategory.contains("Technical")
                              ? "Technical"
                              : "Behavioral",
                      "count": _selectedValue,
                      "score": 8,
                    }),
                  );
  if (response.statusCode == 200) {
    final jsonResp = jsonDecode(response.body);
    final List<dynamic> questions = jsonResp["questions"];
    
    // 3. Pass both username and questions in a Map
    Navigator.pushNamed(
      context,
      '/questions_page',
      arguments: {
        'questions': questions,
        'username': widget.username, // Use the username from the widget
      },
    );
  } else {
                    print("Failed to fetch questions: ${response.statusCode}");
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text("Failed to load questions")),
                    );
                  }
                },




                style: ElevatedButton.styleFrom(
                  backgroundColor: Color.fromARGB(255, 190, 233, 236),
                  padding: EdgeInsets.symmetric(horizontal: 30, vertical: 15),
                ),
                child: Text(
                  'Start Mock Interview',
                  style: TextStyle(
                    fontSize: 22,
                    color: Color.fromARGB(255, 90, 60, 9),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildRadioOption(int value, String text) {
    return InkWell(
      borderRadius: BorderRadius.circular(12),
      onTap: () {
        setState(() {
          _selectedValue = value;
        });
      },
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 6.0),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            Radio(
              value: value,
              groupValue: _selectedValue,
              onChanged: (int? newValue) {
                setState(() {
                  _selectedValue = newValue!;
                });
              },
              fillColor: MaterialStateProperty.all<Color>(
                const Color.fromARGB(255, 97, 97, 97),
              ),
            ),
            Container(
              height: 40.0,
              width: 200.0,
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Center(
                child: Text(
                  text,
                  style: const TextStyle(
                    fontSize: 15,
                    color: Colors.black,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
