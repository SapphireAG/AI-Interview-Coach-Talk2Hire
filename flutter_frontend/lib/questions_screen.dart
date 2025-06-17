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
        title: Center(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(
              10.0,
            ), // Adjust the radius as needed
            child: Image.asset(
              'assets/logo.jpg',
              height: 35.0,
              width: 150.0,
              scale: 1.0,
              fit: BoxFit.cover,
            ),
          ),
        ),
        backgroundColor: Color.fromARGB(255, 91, 106, 234),

        leading: IconButton(
          icon: Icon(Icons.menu),
          iconSize: 30,
          color: Colors.white,
          onPressed: () {
            // Handle menu action
          },
        ),

        actions: [
          IconButton(
            icon: Icon(Icons.account_circle),
            iconSize: 30,
            color: Colors.white,
            onPressed: () {
              // Handle account action
            },
          ),
        ],
      ),

      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,

          children: [
            Container(
              height: 400.0,
              width: 400.0,
              padding: EdgeInsets.all(20.0),
              decoration: BoxDecoration(
                color: Color.fromARGB(255, 91, 106, 234),
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
                    padding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: DropdownButtonHideUnderline(
                      child: DropdownButton<String>(
                        alignment: Alignment.center,
                        value: selectedCategory,
                        dropdownColor:
                            Colors.white, // Background color for dropdown items
                        icon: Icon(
                          Icons.arrow_drop_down,
                          color: Colors.black,
                        ), // Dropdown icon
                        items: [
                          DropdownMenuItem(
                            alignment: Alignment.center,
                            value: "Scenario-Based Questions",
                            child: Text(
                              "Scenario-Based Questions",
                              textAlign: TextAlign.center,
                              style: TextStyle(color: Colors.black),
                            ),
                          ),
                          DropdownMenuItem(
                            alignment: Alignment.center,
                            value: "Technical Questions",
                            child: Text(
                              textAlign: TextAlign.center,
                              "Technical Questions",
                              style: TextStyle(color: Colors.black),
                            ),
                          ),
                          DropdownMenuItem(
                            alignment: Alignment.center,
                            value: "Personal Questions",
                            child: Text(
                              "Personal Questions",
                              textAlign: TextAlign.center,
                              style: TextStyle(color: Colors.black),
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

                  Text(
                    'Select the number of Questions',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  SizedBox(height: 20),
                  _buildRadioOption(5, "5 Questions"),
                  SizedBox(height: 20),

                  _buildRadioOption(10, "10 Questions"),
                  SizedBox(height: 20),
                  _buildRadioOption(15, "15 Questions"),

                  SizedBox(height: 20), // Spacing before dropdown box
                ],
              ),
            ),

            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                _startMock();
                // Navigator.pushNamed(context, '/nextPage');
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Color.fromARGB(255, 91, 106, 234),
                padding: EdgeInsets.symmetric(horizontal: 50, vertical: 15),
              ),
              child: Text(
                'Start Mock Interview',
                style: TextStyle(
                  fontSize: 24,
                  color: Colors.white,
                  fontStyle: FontStyle.normal,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRadioOption(int value, String text) {
     return Row(
        mainAxisAlignment:
            MainAxisAlignment.start, 
        children: [
          Radio(
            autofocus: false,
            fillColor: WidgetStateProperty.all<Color>(
              Colors.white,
            ), // White color for the radio button
            value: value,
            groupValue: _selectedValue,
            onChanged: (int? newValue) {
              setState(() {
                _selectedValue = newValue!;
              });
            },
          ),
          Container(
            height: 40.0,
            width: 200.0,
            padding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.white, // White background for the box
              borderRadius: BorderRadius.circular(10), // Rounded corners
            ),
            child: Center(
              // ✅ Ensures text inside box is centered
              child: Text(
                text,
                style: TextStyle(
                  fontSize: 15,
                  color: Colors.black,
                  fontWeight: FontWeight.normal,
                ),
              ),
            ),
          ),
        ],
      
    );
  }
}
