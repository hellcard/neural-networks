#include <iostream>

int dot(int arr1[],int arr2[],int size);
int step_func(int x);
int perceptron_output(int weights[],float bias,int nums[],int size);
void replace_el(int arr[], int el1, int el2);
void cout_nums(int nums[], int size);

int main(){

        // logic gate AND

	int weights[]={2,2};
	float bias=-3;
	int nums[]={1,1};

        std::cout<<"logic gate AND:"<<std::endl;

        cout_nums(nums,2);
	std::cout<<perceptron_output(weights,bias,nums,2)<<std::endl;

	replace_el(nums,0,1);
	cout_nums(nums,2);
	std::cout<<perceptron_output(weights,bias,nums,2)<<std::endl;

	replace_el(nums,1,0);
	cout_nums(nums,2);
	std::cout<<perceptron_output(weights,bias,nums,2)<<std::endl;

	replace_el(nums,0,0);
	cout_nums(nums,2);
	std::cout<<perceptron_output(weights,bias,nums,2)<<std::endl;

	std::cout<<"---"<<std::endl;

	// logic gate OR

	bias=-1;

	std::cout<<"logic gate OR:"<<std::endl;

	replace_el(nums,1,1);
	cout_nums(nums,2);
	std::cout<<perceptron_output(weights,bias,nums,2)<<std::endl;

	replace_el(nums,0,1);
	cout_nums(nums,2);
	std::cout<<perceptron_output(weights,bias,nums,2)<<std::endl;

	replace_el(nums,1,0);
	cout_nums(nums,2);
	std::cout<<perceptron_output(weights,bias,nums,2)<<std::endl;

	replace_el(nums,0,0);
	cout_nums(nums,2);
	std::cout<<perceptron_output(weights,bias,nums,2)<<std::endl;

	std::cout<<"---"<<std::endl;

	// logic gate NOT

	replace_el(weights,-2,0);
	bias=1;

	std::cout<<"logis gate NOT:"<<std::endl;

	replace_el(nums,1,0);
	cout_nums(nums,1);
	std::cout<<perceptron_output(weights,bias,nums,1)<<std::endl;

	replace_el(nums,0,0);
	cout_nums(nums,1);
	std::cout<<perceptron_output(weights,bias,nums,1)<<std::endl;

	return 0;
}
int dot(int arr1[],int arr2[],int size){
	int mul[size];
	for(int i=0;i<size;i++){
		int res=arr1[i]*arr2[i];
		mul[i]=res;
	}
	int res=0;
	for(int i=0;i<size;i++){
		res+=mul[i];
	}
	return res;
}
int step_func(int x){
    return x>=0 ? 1 : 0;
}

int perceptron_output(int weights[],float bias,int nums[], int size){
	int res=dot(weights,nums,size)+bias;
	return step_func(res);
}

void replace_el(int arr[], int el1, int el2){
    arr[0]=el1;
    arr[1]=el2;
}

void cout_nums(int nums[],int size){
    for(int i=0;i<size;i++){
        std::cout<<nums[i]<<" ";
    }
    std::cout<<"=> ";
}
