import 'package:flutter/material.dart';

class QuestionsPage extends StatefulWidget {
  const QuestionsPage({super.key});
  @override
  State<QuestionsPage> createState() => _QuestionsPageState();
}

String selectedCategory = "Scenario-Based Questions";
int _selectedValue = 10; // Default selection

void _startMock() {
  // Implement your mock start logic here
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
              // Adjust the radius as needed
              child: Image.asset(
                'assets/NEW_logo.png',
                height: 80.0,
                width: 150.0,
                //scale: 1.0,
                fit: BoxFit.contain,
              ),
            ),
          ),
        ),
        backgroundColor: Color.fromARGB(255, 249, 250, 251),

        leading: Builder(
          builder:
              (context) => IconButton(
                icon: Icon(Icons.menu, color: Colors.black38, size: 35),
                iconSize: 40,
                onPressed: () {
                  Scaffold.of(context).openDrawer(); // <-- THIS works now
                },
              ),
        ),

        actions: [
          IconButton(
            icon: Icon(Icons.account_circle),
            iconSize: 40,
            color: Colors.black38,
            onPressed: () {
              // Handle account action
            },
          ),
        ],
      ),

      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            Container(
              height: 100, // Reduced height from default 200
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
                Navigator.pop(context); // Close the drawer
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
                    // Dropdown Box
                    Container(
                      height: 50.0,
                      width: double.infinity,
                      padding: EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 10,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: DropdownButtonHideUnderline(
                        child: DropdownButton<String>(
                          borderRadius: BorderRadius.circular(12),
                          alignment: Alignment.center,
                          value: selectedCategory,
                          dropdownColor:
                              Colors
                                  .white, // Background color for dropdown items
                          icon: Icon(
                            Icons.arrow_drop_down,
                            color: Colors.black,
                          ), // Dropdown icon
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
                        Icon(
                          Icons.question_answer_outlined,
                          color: Colors.black54,
                        ),
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

                    SizedBox(height: 10), // Spacing before dropdown box
                  ],
                ),
              ),

              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  _startMock();
                  Navigator.pushNamed(context, '/questions_page');
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
                    fontStyle: FontStyle.normal,
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
              fillColor: WidgetStateProperty.all<Color>(
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
